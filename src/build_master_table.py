"""
Build the consolidated patient master table (data/patients_master.xlsx).

Sources
-------
1. data/sonya_data.xlsx (Лист2)      -> Department 2 (MG-2-...), used as-is (incl. spectra).
2. data/raw/2026 patients/*.txt      -> Department 1, new patients (MG-1-...), device export,
                                         two template variants co-exist; parsed via a
                                         name-anchored resync scan (robust to stray blank
                                         fields inserted by the device firmware).
3. data/raw_spectra_to_MG168.xlsx +
   data/Old ID to MG.xlsx +
   data/raw/Parkinson first devise/*  -> Department 1, OLD patients (MG-... -> MG-1-...).
                                         Clinical fields are very sparse (only what the old
                                         table recorded); equipment string is recovered from
                                         the matching raw .txt file when found.

Spectra (wavelength columns, 68-579) are populated for all three sources. Sonya's (dept2)
spectra are copied as-is (they define the target grid). Department-1 spectra (old, from
raw_spectra_to_MG168's 221-point grid, and new, from each raw .txt file's native 512-point
grid) are interpolated onto Sonya's grid via preprocessing.interpolate_spectra_to_grid
(cubic spline; points outside a source's observed wavelength range are left blank rather
than extrapolated). See preprocessing.read_raw_spectrum_file for the raw-file reader.

Per user decision (2026-08-19 clarification round):
- GUIDs ('ID исследования') are generated (uuid4) only for OLD department-1 rows. New
  department-1 rows already carry a real device-issued GUID and keep it.
- For duplicate/near-duplicate raw files (.txt, .txt.txt, .txt.txt.txt) the content
  ('ID исследования' GUID) is authoritative, not the file name; duplicates are merged
  (coalesced) into one row per GUID.

Output: data/patients_master.xlsx (single header row + data), plus a text report of
match rates / dropped fields / ambiguities: data/patients_master_report.txt
"""
from __future__ import annotations

import glob
import os
import re
import unicodedata
import uuid
from collections import defaultdict

import numpy as np
import openpyxl
import pandas as pd

from preprocessing import read_raw_spectrum_file, interpolate_spectra_to_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

SONYA_PATH = os.path.join(DATA, "sonya_data.xlsx")
DEPT1_NEW_DIR = os.path.join(DATA, "raw", "2026 patients")
OLD_SPECTRA_PATH = os.path.join(DATA, "raw_spectra_to_MG168.xlsx")
OLD_ID_MAP_PATH = os.path.join(DATA, "Old ID to MG.xlsx")
PARK_FIRST_DIR = os.path.join(DATA, "raw", "Parkinson first devise")

OUT_XLSX = os.path.join(DATA, "patients_master.xlsx")
OUT_REPORT = os.path.join(DATA, "patients_master_report.txt")

N_HEADER_COLS = 580  # full Sonya header width (clinical block 0-67 + 512 spectral cols)
SPECTRA_START = 68
SPECTRA_END = N_HEADER_COLS  # bounded end for spectra slicing; rows may grow past
                              # this (extra columns appended after), so this must
                              # stay an explicit bound, never an open-ended `[68:]`

# Two extra columns appended after the spectra block (indices 580, 581):
# Parkinson's disease stage and yes/no flag. Kept as an append-only extension
# so the 512-wide spectra slicing above (indices 68-579) is untouched.
PD_STAGE_HEADER = "Стадия болезни Паркинсона (-1 - нет БП, 0-5 - стадия Хен-Яра)"
PD_FLAG_HEADER = "Болезнь Паркинсона (да|нет)"
PD_NO_DISEASE_STAGE = -1  # not 0: stage 0 (subclinical) is itself a valid Hoehn-Yahr value

report_lines: list[str] = []


def log(msg: str) -> None:
    report_lines.append(msg)
    print(msg)


# "мса" (multiple system atrophy) is a distinct atypical-parkinsonism syndrome,
# not idiopathic Parkinson's disease (БП) -> deliberately NOT matched here.
_PD_KEYWORD_RE = re.compile(r"\bбп\b|паркинсон", re.IGNORECASE)
_PD_STAGE_RE = re.compile(r"([0-5])\s*ст|стад\w*\D{0,3}([0-5])", re.IGNORECASE)


def parse_pd_from_disease_text(text) -> tuple[bool, int | None]:
    """Best-effort parse of ('has Parkinson's disease', stage 0-5 or None) out
    of a free-text diagnosis field (column 10, 'Основное неврологическое
    заболевание'). Returns (False, None) if no Parkinson's mention is found."""
    if not text or not isinstance(text, str):
        return False, None
    has_pd = bool(_PD_KEYWORD_RE.search(text))
    if not has_pd:
        return False, None
    m = _PD_STAGE_RE.search(text)
    stage = None
    if m:
        stage = int(m.group(1) or m.group(2))
    return True, stage


def pd_columns_from_disease_text(disease_text) -> tuple[int | None, str | None]:
    """(stage, flag) for the two appended PD columns, from free-text diagnosis.
    flag='да' + stage=0-5 (or None if unspecified) when Parkinson's is mentioned;
    flag='нет' + stage=-1 when a disease field was present but didn't mention it;
    (None, None) when there was no disease text to judge from at all."""
    has_pd, stage = parse_pd_from_disease_text(disease_text)
    if has_pd:
        return stage, "да"
    if isinstance(disease_text, str) and disease_text.strip():
        return PD_NO_DISEASE_STAGE, "нет"
    return None, None


# ---------------------------------------------------------------------------
# 1. Sonya (Department 2) — used verbatim
# ---------------------------------------------------------------------------

def load_sonya():
    df = pd.read_excel(SONYA_PATH, sheet_name="Лист2", header=None)
    header = df.iloc[0, :].tolist()
    assert len(header) == N_HEADER_COLS, len(header)
    data_rows = df.iloc[2:, :].values.tolist()
    log(f"[Sonya/dept2] header cols={len(header)} rows={len(data_rows)}")

    for row in data_rows:
        stage, flag = pd_columns_from_disease_text(row[10])
        row.append(stage)
        row.append(flag)

    target_wavelengths = np.array(
        [float(str(x).replace(",", ".")) for x in header[SPECTRA_START:]]
    )
    return header, data_rows, target_wavelengths


# ---------------------------------------------------------------------------
# 2. Department 1, NEW patients (data/raw/2026 patients)
# ---------------------------------------------------------------------------

# raw field label -> target column index in the master header, or None to drop
# (order matters only for disambiguating duplicated labels: first occurrence in
#  the token stream consumes the first tuple with that label, etc.)
FIELD_SPEC: list[tuple[str, int | None]] = [
    # hand is taken from the filename suffix (Л/П), not this free-text field:
    # it frequently disagrees with the filename and the filename is the more
    # reliable signal (device-generated, not manually typed).
    ("рука", None),
    ("Возраст пациента (лет)", 5),
    ("Пол пациента (м|ж)", 6),
    ("Вес кг", 8),
    ("Рост, м", 9),
    ("Основное неврологическое заболевание", 10),
    ("Сопутствующие заболевания", 12),
    ("Курение", 15),
    ("СД2", 17),
    ("Коды заболевания по МКБ11\\стадия\\степень тяжести - длительность лет (данные через запятую)", None),
    ("Терапия (Препарат - дозировка, данные через запятую)", None),
    ("ОAK (Дата забора крови)", 18),
    ("Гемоглобин  г/л, ", 19),
    ("Эритроциты  /л, ", 20),
    ("Нейтрофилы %, ", 21),
    ("Цветовой показатель", 22),
    ("Тромбоциты /л,", 23),
    ("Лейкоциты /л, ", 24),
    ("Эозинофилы %, ", 25),
    ("Базофилы %, ", 26),
    # 'Анализ мочи' (col 27) holds a lab value (urine specific gravity, ~1.00-1.03)
    # in Sonya's data, not a date -> the raw collection date has no matching column.
    ("ОАМ (Дата забора мочи)", None),
    ("Кетоновые тела ", None),
    ("Белок  г/л, ", None),
    ("Анализ мочи", None),
    ("Хинолиновая кислота", 28),
    ("Кинуреновая кислота", 29),
    ("БХА (Дата забора крови)", 30),
    ("Глюкоза ммоль/л, ", 31),
    ("Инсулин мкМЕ/мл", 32),
    ("С-пептид нг/мл или Инсулин", 32),
    ("HbA1c %", 33),
    ("Лактат ммоль/л", 34),
    ("Мочевая кислота мкм/л", 35),
    ("Мочевина", 36),
    ("Креатинин", 37),
    ("Липидный спектр (Дата забора крови)", 38),
    ("Общий холестерин ммоль/л", 39),
    ("Триглицериды ммоль/л", 40),
    ("Холестерин ЛПНП ммоль/л", 41),
    ("Холестерин ЛПВП ммоль/л", 42),
    ("Холестерин ЛПОНП ммоль/л", None),
    ("Холестерин, (не-ЛПВП) ммоль/л", None),
    ("Липопротеин а,", 43),
    ("Липопротеин а", 43),
    ("ИФА крови (Дата забора крови)", 44),
    ("sRAGE пг/мл", 45),               # blood
    ("AGEs  нг/мл", 46),
    ("Интерлейкин-1 пг/мл ", 47),      # blood
    ("Кинуренин  мкМ", 48),
    ("Альфа-синуклеин (SNCA) мкг/л", None),  # blood (no matching column)
    ("ИФА ЦСЖ (Дата забора ЦСЖ)", 49),
    ("sRAGE пг/мл", 50),               # CSF
    ("Интерлейкин-1 пг/мл ", 51),      # CSF
    ("Альфа-синуклеин (SNCA) мкг/л", None),  # CSF (no matching column)
    ("Иммуноблоттинг альфа-синуклеина в экзосомах крови мкг/л", 52),
    ("RT-QuiC экзосом крови (Дата забора крови)", 53),
    ("T0.8-T0.2 час ", 54),            # exosome
    ("T0.5 час", 55),                  # exosome
    ("RT-QuiC ЦСЖ (Дата забора ЦСЖ)", 56),
    ("T0.8-T0.2 час ", 57),            # CSF
    ("T0.5 час", 58),                  # CSF
    ("Расширенный анализ мочи на органические кислоты 60 показателей HELIX 06-224 (Показатель-значение)", 59),
]

# labels handled specially outside FIELD_SPEC (identity / GUID columns)
LABEL_ID_STUDY = "ID исследования"
LABEL_DATETIME = "Дата.Время"
LABEL_PATIENT_ID = "ID пациента"

_PLACEHOLDER_VALUES = {"дата", "-", "—", "нет данных"}
# literal template boilerplate left un-replaced by operators (not real patient data)
_PLACEHOLDER_PREFIXES = ("Молочная кислота -1 ммоль/л, Кинуреновая кислота",)


def _sanitize_value(v: str) -> str:
    v = v.strip()
    if v.lower() in _PLACEHOLDER_VALUES:
        return ""
    if v.startswith(_PLACEHOLDER_PREFIXES):
        return ""
    return v


def _build_label_queues():
    """label (stripped) -> list of column indices (or None) in encounter order, consumed via pop(0)."""
    queues: dict[str, list[int | None]] = defaultdict(list)
    for label, col in FIELD_SPEC:
        queues[label.strip()].append(col)
    return queues


ALL_KNOWN_LABELS = {LABEL_ID_STUDY, LABEL_DATETIME, LABEL_PATIENT_ID} | {l.strip() for l, _ in FIELD_SPEC}


def parse_dept1_line(line: str) -> dict[int, str]:
    """Resync-scan the semicolon-delimited metadata line into {col_idx: value}."""
    tokens = line.rstrip("\n\r").split(";")
    queues = _build_label_queues()

    # locate positions of all recognized key tokens, in order
    key_positions = [i for i, t in enumerate(tokens) if t.strip() in ALL_KNOWN_LABELS]

    result: dict[int, str] = {}
    guid = patient_id = dt = ""

    for idx, pos in enumerate(key_positions):
        label = tokens[pos].strip()
        next_pos = key_positions[idx + 1] if idx + 1 < len(key_positions) else len(tokens)
        between = tokens[pos + 1:next_pos]
        value = ""
        for t in between:
            if t.strip() != "":
                value = t.strip()
        value = _sanitize_value(value)

        if label == LABEL_ID_STUDY:
            guid = value
            continue
        if label == LABEL_DATETIME:
            dt = value
            continue
        if label == LABEL_PATIENT_ID:
            patient_id = value
            continue

        if label in queues and queues[label]:
            col = queues[label].pop(0)
            if col is not None and value != "":
                # don't let an empty later duplicate stomp a real value collected earlier
                if col not in result or result[col] == "":
                    result[col] = value

    result["_guid"] = guid
    result["_dt"] = dt
    result["_pid"] = patient_id
    return result


def _hand_from_filename(fname: str) -> str:
    if re.search(r"[(\-]?Л[)\-]?", fname):
        return "Левая"
    if re.search(r"[(\-]?П[)\-]?", fname):
        return "Правая"
    return ""


def parse_dept1_new(target_wavelengths: np.ndarray):
    files = sorted(
        f for f in glob.glob(os.path.join(DEPT1_NEW_DIR, "*.txt"))
        if os.path.isfile(f)
    )
    log(f"[dept1/new] {len(files)} .txt files found")

    by_guid: dict[str, dict] = {}
    mismatches = []

    for path in files:
        fname = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                first_line = fh.readline()
                second_line = fh.readline()
                fh.readline()  # points-per-spectrum count, unused here
                pulses_line = fh.readline()
        except UnicodeDecodeError:
            log(f"  ! could not decode {fname} as utf-8-sig, skipping")
            continue

        try:
            n_pulses = int(pulses_line.strip())
        except ValueError:
            n_pulses = 8

        parsed = parse_dept1_line(first_line)
        guid = parsed.pop("_guid")
        dt = parsed.pop("_dt")
        pid = parsed.pop("_pid")
        pid = re.sub(r"[\s\-\(]*[ЛПлп]\)?$", "", pid).strip()
        equipment = second_line.strip()

        if not guid:
            log(f"  ! {fname}: no GUID found, skipping")
            continue

        digit_groups = re.findall(r"\d+", fname.split(".")[0])
        fname_pid = f"MG-1-{digit_groups[-1]}" if len(digit_groups) >= 2 else None

        if not pid or not re.match(r"MG-1-\d+$", pid):
            # content ID missing/truncated/malformed -> fall back to filename-derived ID
            if fname_pid:
                mismatches.append((fname, f"content={pid!r} -> using filename-derived {fname_pid!r}"))
                pid = fname_pid
        elif fname_pid and fname_pid != pid:
            mismatches.append((fname, f"content={pid!r} (kept; filename suggested {fname_pid!r})"))

        hand = _hand_from_filename(fname)
        mtime = os.path.getmtime(path)

        if guid not in by_guid:
            by_guid[guid] = {
                "pid": pid,
                "dt": dt,
                "hand": hand,
                "equipment": equipment,
                "n_pulses": n_pulses,
                "fields": {},
                "files": [],
                "_mtime": mtime,
                "spectrum_path": path,
            }
        entry = by_guid[guid]
        entry["files"].append(fname)
        if mtime >= entry["_mtime"]:
            entry["_mtime"] = mtime
            if not entry["pid"]:
                entry["pid"] = pid
            if not entry["hand"]:
                entry["hand"] = hand
            if equipment:
                entry["equipment"] = equipment
            entry["n_pulses"] = n_pulses
            entry["spectrum_path"] = path
        for col, val in parsed.items():
            if val == "":
                continue
            # later file wins ties, but never overwrite a real value with blank
            entry["fields"][col] = val

    if mismatches:
        log(f"  ! {len(mismatches)} filename/content ID mismatches (content wins, filenames listed):")
        for fname, pid in mismatches:
            log(f"      {fname} -> internal ID пациента = {pid!r}")

    log(f"[dept1/new] merged into {len(by_guid)} unique studies (by GUID)")
    total_pulse_rows = sum(e["n_pulses"] for e in by_guid.values())
    log(f"[dept1/new] expanding to {total_pulse_rows} rows (one per pulse, per Sonya's convention)")

    # native wavelength grid of the 2026-patients device vs Sonya's (target) grid
    native_wl = None
    grid_offsets = []
    total_nan = total_cells = 0

    rows = []
    for guid, entry in sorted(by_guid.items(), key=lambda kv: kv[1]["pid"]):
        base = [None] * N_HEADER_COLS
        base[0] = entry["equipment"]
        base[1] = "; ".join(sorted(set(entry["files"])))
        base[2] = guid
        base[3] = entry["dt"]
        base[4] = entry["pid"]
        if entry["hand"]:
            base[7] = entry["hand"]
        for col, val in entry["fields"].items():
            base[col] = val
        stage, flag = pd_columns_from_disease_text(base[10])
        base.append(stage)
        base.append(flag)

        try:
            spec = read_raw_spectrum_file(entry["spectrum_path"])
            if native_wl is None:
                native_wl = spec["wavelengths"]
                grid_offsets.append(float(np.mean(native_wl - target_wavelengths)))
            interp = interpolate_spectra_to_grid(spec["wavelengths"], spec["spectra"], target_wavelengths)
            total_nan += int(np.isnan(interp).sum())
            total_cells += interp.size
        except Exception as e:
            log(f"  ! {entry['spectrum_path']}: failed to read/interpolate spectrum ({e})")
            interp = None

        n_pulses = interp.shape[0] if interp is not None else entry["n_pulses"]
        for pulse_idx in range(n_pulses):
            row = list(base)
            row[SPECTRA_START - 1] = pulse_idx  # 'Pulses|Wavelenghts' column
            if interp is not None:
                row[SPECTRA_START:SPECTRA_END] = [
                    None if np.isnan(v) else float(v) for v in interp[pulse_idx]
                ]
            rows.append(row)

    if native_wl is not None:
        log(f"[dept1/new] native wavelength grid: {len(native_wl)} points, "
            f"{native_wl[0]:.2f}-{native_wl[-1]:.2f} nm vs Sonya's target grid "
            f"{target_wavelengths[0]:.2f}-{target_wavelengths[-1]:.2f} nm "
            f"(mean offset {grid_offsets[0]:+.2f} nm) -> MISMATCH, interpolated to target grid")
        pct = 100 * total_nan / total_cells if total_cells else 0
        log(f"[dept1/new] interpolation coverage: {total_nan}/{total_cells} cells "
            f"({pct:.1f}%) fall outside the native range and are left blank (no extrapolation)")

    return rows


# ---------------------------------------------------------------------------
# 3. Department 1, OLD patients (raw_spectra_to_MG168 + Old ID to MG + raw txt)
# ---------------------------------------------------------------------------

# Cyrillic/Latin homoglyphs that appear interchangeably in these filenames/ID codes
_HOMOGLYPHS = str.maketrans({
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h",
    "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
})


def _normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\.txt(\.csv)?$", "", s)
    s = re.sub(r"_sp$", "", s)
    s = re.sub(r"[\(\)]", "", s)
    s = re.sub(r"[лп]$", "", s)  # trailing hand marker after paren-strip, e.g. "name-л"
    s = s.translate(_HOMOGLYPHS)
    s = re.sub(r"[^0-9a-zа-яё]+", "", s)
    return s


def _index_park_first_files() -> dict[str, str]:
    """normalized-stem -> path, over all .txt under Parkinson first devise (root+RAW+spectres)."""
    idx: dict[str, str] = {}
    paths = glob.glob(os.path.join(PARK_FIRST_DIR, "**", "*.txt"), recursive=True)
    paths += glob.glob(os.path.join(PARK_FIRST_DIR, "**", "*.csv"), recursive=True)
    for path in paths:
        fname = os.path.basename(path)
        if fname.lower() in ("desktop.ini",) or fname == "tocsv.bat":
            continue
        if fname.upper().endswith("_SP.TXT"):
            # processed/aggregated spectrum export, different header, no equipment line
            continue
        key = _normalize_name(fname)
        # prefer shallower (root-level) files over nested RAW/spectres duplicates
        if key not in idx or os.path.dirname(path).count(os.sep) < os.path.dirname(idx[key]).count(os.sep):
            idx[key] = path
    return idx


def _read_equipment_line(path: str) -> str:
    for enc in ("utf-16-le", "utf-8-sig", "cp1251"):
        try:
            with open(path, "r", encoding=enc) as fh:
                fh.readline()
                line = fh.readline().strip()
            if line and re.search(r"(подключ|spectomet|spectromet|контроллер)", line, re.IGNORECASE):
                return line
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def _find_constant_old_equipment() -> str:
    """The old device is the same single spectrometer for the whole old dataset
    (spot-checked equipment lines across Группа сравнения / Контр группа /
    Паркинсон пациенты are byte-identical) -> resolve it once and reuse."""
    file_index = _index_park_first_files()
    for path in file_index.values():
        line = _read_equipment_line(path)
        if line:
            return line
    return ""


def parse_dept1_old(target_wavelengths: np.ndarray):
    raw = pd.read_excel(OLD_SPECTRA_PATH, sheet_name="Sheet1", header=None)
    old_wavelengths = np.array([float(x) for x in raw.iloc[0, 8:]])
    spec = raw.iloc[1:].reset_index(drop=True)
    spec.columns = raw.iloc[0]
    spec["MG ID"] = spec["MG ID"].ffill()
    # block index: increments at every row where 'hand' was explicitly given
    # (i.e. the start of a new file/measurement within a patient)
    spec["_block"] = spec["hand"].notna().groupby(spec["MG ID"]).cumsum()
    spec["hand"] = spec.groupby("MG ID")["hand"].ffill()

    agg = spec.groupby("MG ID", sort=False).first()[["PARKINSON", "GROUP", "SEX", "AGE", "STAGE"]]
    log(f"[dept1/old] {len(agg)} unique old MG-x patients in raw_spectra_to_MG168, "
        f"{len(spec)} spectrum rows total")

    id_map = pd.read_excel(OLD_ID_MAP_PATH)
    id_map = id_map.set_index("ID MG")

    equipment = _find_constant_old_equipment()
    log(f"[dept1/old] equipment (constant across the whole old dataset): {equipment!r}")

    sex_map = {"male": "м", "female": "ж"}
    hand_map = {"left": "Левая", "right": "Правая"}
    hand_suffix = {"left": "L", "right": "R"}

    rows = []
    raw_spectra = []
    guid_cache: dict[tuple, str] = {}

    for mg_id, row_data in agg.iterrows():
        m = re.match(r"MG-(\d+)", str(mg_id))
        if not m:
            continue
        new_id = f"MG-1-{m.group(1)}"

        patient_rows = spec[spec["MG ID"] == mg_id]
        for _, r in patient_rows.iterrows():
            hand = r.get("hand")
            hand = hand.strip().lower() if isinstance(hand, str) else ""
            block_key = (mg_id, r["_block"])
            if block_key not in guid_cache:
                guid_cache[block_key] = str(uuid.uuid4())
            guid = guid_cache[block_key]

            suffix = hand_suffix.get(hand, "")
            file_label = f"{new_id}_{suffix}" if suffix else new_id

            try:
                pulse_idx = int(r["No spec"]) - 1
            except (ValueError, TypeError):
                pulse_idx = 0

            row = [None] * N_HEADER_COLS
            row[0] = equipment
            row[1] = file_label
            row[2] = guid
            row[4] = new_id
            row[SPECTRA_START - 1] = pulse_idx
            if hand:
                row[7] = hand_map.get(hand, hand)
            _fill_old_metadata(row, row_data, sex_map)
            pd_stage, pd_flag = _old_pd_stage(row_data)
            row.append(pd_stage)
            row.append(pd_flag)
            rows.append(row)
            raw_spectra.append(r.iloc[8:8 + len(old_wavelengths)].to_numpy(dtype=float))

    log(f"[dept1/old] emitted {len(rows)} rows (one per pulse, per Sonya's convention)")
    if _out_of_range_pd_stages:
        distinct = sorted(set(_out_of_range_pd_stages))
        log(f"[dept1/old] {len(distinct)} patient(s) have STAGE > 5 in the source table "
            f"(outside the Hoehn-Yahr 0-5 range; kept the true source value, not clipped): {distinct}")

    raw_spectra = np.array(raw_spectra)
    interp = interpolate_spectra_to_grid(old_wavelengths, raw_spectra, target_wavelengths)
    n_nan = int(np.isnan(interp).sum())
    log(f"[dept1/old] native wavelength grid: {len(old_wavelengths)} points, "
        f"{old_wavelengths[0]:.2f}-{old_wavelengths[-1]:.2f} nm vs Sonya's target grid "
        f"{target_wavelengths[0]:.2f}-{target_wavelengths[-1]:.2f} nm -> narrower range, "
        f"interpolated (no extrapolation) to target grid")
    log(f"[dept1/old] interpolation coverage: {n_nan}/{interp.size} cells "
        f"({100 * n_nan / interp.size:.1f}%) fall outside the native range and are left blank")

    for row, spectrum in zip(rows, interp):
        row[SPECTRA_START:SPECTRA_END] = [None if np.isnan(v) else float(v) for v in spectrum]

    return rows


_out_of_range_pd_stages: list[tuple[str, int]] = []


def _old_pd_stage(row_data) -> tuple[int | None, str | None]:
    """Authoritative PD stage/flag for old dept1 rows, from raw_spectra_to_MG168's
    own PARKINSON/STAGE columns (not text-parsed)."""
    parkinson = row_data.get("PARKINSON")
    if not isinstance(parkinson, str):
        return None, None
    if parkinson.strip().upper() == "NO":
        return PD_NO_DISEASE_STAGE, "нет"
    stage_raw = row_data.get("STAGE")
    try:
        stage = int(stage_raw)
    except (TypeError, ValueError):
        return None, "да"
    if stage < 0:
        return None, "да"  # STAGE=-1 sentinel: PD present but stage not recorded
    if stage > 5:
        _out_of_range_pd_stages.append((str(row_data.name), stage))
    return stage, "да"


def _fill_old_metadata(row, row_data, sex_map):
    age = row_data.get("AGE")
    if pd.notna(age):
        row[5] = age
    sex = row_data.get("SEX")
    if isinstance(sex, str):
        row[6] = sex_map.get(sex.strip().lower(), sex)
    group = row_data.get("GROUP")
    if isinstance(group, str) and group.strip():
        row[10] = group.strip()


# ---------------------------------------------------------------------------
# cross-source normalization: make every column self-consistent in type,
# format and units regardless of which source (Sonya / new dept1 / old dept1)
# a row came from.
# ---------------------------------------------------------------------------

DATE_COLUMNS = [18, 30, 38, 44, 49, 53, 56]  # "...(Дата забора крови/ЦСЖ)" group headers

# col -> (exponent, unit note); suffix moved out of the cell into the header text
UNIT_SUFFIX_COLUMNS = {20: "12", 23: "9", 24: "9"}

NUMERIC_COLUMNS = (
    [5, 8, 9, 19, 21, 22, 25, 26]
    + [31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43]
    + [54, 55, 57, 58]
    + [60, 61, 62, 63, 64, 65, 66, 67]
    + list(UNIT_SUFFIX_COLUMNS)
)

HAND_NORMALIZE = {
    "l": "Левая", "л": "Левая", "left": "Левая", "левая": "Левая",
    "r": "Правая", "п": "Правая", "right": "Правая", "правая": "Правая",
}

# cells that are literally invalid for their column's declared format
# (confirmed present verbatim in the source, not a parsing artifact)
INVALID_CELL_VALUES = {17: {"сд1"}, 52: {"**"}}


def _to_native(v):
    """numpy scalar -> native Python scalar, otherwise unchanged."""
    if hasattr(v, "item") and not isinstance(v, (pd.Timestamp,)):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return v
    return v


def _to_date(v):
    v = _to_native(v)
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    import datetime as _dt
    if isinstance(v, _dt.date):
        return v
    s = str(v).strip().rstrip(",").strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return v  # unparseable -> leave untouched rather than lose data


_TRAILING_UNIT_WORDS = ("см", "мм", "кг")


def _to_number(v):
    """Best-effort numeric coercion that never guesses: a value only becomes a
    number if what's left after stripping a known unit suffix is unambiguously
    numeric. Anything else (stray text, a value-less '*10(9)' unit template,
    a misplaced free-text note) is left untouched and reported, not mangled."""
    v = _to_native(v)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and pd.isna(v) else v
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None

    # strip a trailing '*10^(N)' / '*10(N)' exponent suffix (moved to the header)
    m = re.match(r"^(.*?)\s*\*\s*10\s*\^?\(?\s*-?\d+\s*\)?\s*$", s)
    core = m.group(1).strip() if m else s
    if core == "":
        return None  # e.g. a bare '*10(9)' template with no number ever typed

    core = re.sub(
        rf"\s*(?:{'|'.join(_TRAILING_UNIT_WORDS)})\.?\s*$", "", core, flags=re.IGNORECASE
    )
    core = core.replace(",", ".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", core):
        return v  # contains letters/garbage -> not unambiguously numeric, leave as-is
    f = float(core)
    return int(f) if f.is_integer() else f


def normalize_rows(header, rows):
    header = list(header)
    for col, exp in UNIT_SUFFIX_COLUMNS.items():
        header[col] = f"{header[col].rstrip()} (×10^{exp})"

    unparsed = []
    for row in rows:
        for col in DATE_COLUMNS:
            row[col] = _to_date(row[col])
        for col in NUMERIC_COLUMNS:
            before = row[col]
            after = _to_number(before)
            if before is not None and after == before and isinstance(after, str):
                unparsed.append((col, before))
            row[col] = after
        hand = row[7]
        if isinstance(hand, str) and hand.strip():
            row[7] = HAND_NORMALIZE.get(hand.strip().lower(), hand.strip())
        for col, bad_values in INVALID_CELL_VALUES.items():
            if isinstance(row[col], str) and row[col].strip().lower() in bad_values:
                row[col] = None

    if unparsed:
        log(f"[normalize] {len(unparsed)} numeric cells left as-is (could not parse):")
        for col, val in unparsed[:20]:
            log(f"      col {col} ({header[col]!r}): {val!r}")

    return header, rows


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def write_output(header, sonya_rows, dept1_new_rows, dept1_old_rows):
    header = list(header) + [PD_STAGE_HEADER, PD_FLAG_HEADER]
    all_rows = dept1_old_rows + dept1_new_rows + sonya_rows
    for r in all_rows:
        assert len(r) == len(header), f"row width {len(r)} != header width {len(header)}"
    header, all_rows = normalize_rows(header, all_rows)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "patients"
    ws.append(header)
    for r in all_rows:
        ws.append(r)
    wb.save(OUT_XLSX)
    log(f"[write] {OUT_XLSX}: {len(all_rows)} data rows "
        f"({len(dept1_old_rows)} old dept1 + {len(dept1_new_rows)} new dept1 + {len(sonya_rows)} dept2)")


def main():
    header, sonya_rows, target_wavelengths = load_sonya()
    dept1_new_rows = parse_dept1_new(target_wavelengths)
    dept1_old_rows = parse_dept1_old(target_wavelengths)
    write_output(header, sonya_rows, dept1_new_rows, dept1_old_rows)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"[report] {OUT_REPORT}")


if __name__ == "__main__":
    main()
