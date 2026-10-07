"""Config dataclasses and schema validation for the spectra ML pipeline.

Stage 1 of current_tasks/build_project_and_exp.md: nothing here may depend on
`target` in a way that leaks into QC/preprocessing later, and patient_id must
be a hard requirement (missing patient_id => stop with a clear error).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class DataConfig:
    data_path: str
    sheet_name: str
    patient_id_col: str
    legacy_mg_id_col: str
    instrument_col: str
    target_col: str
    target_map: dict[str, int]
    wavelength_header_row_hint: str
    wavelength_min_nm: float
    wavelength_max_nm: float
    random_seed: int
    # Identity-merge columns (spectra_ml.identity): used to detect same-
    # session, two-hand recordings that were assigned separate GUIDs (see
    # configs/data.yaml). Optional/None for backward compatibility with any
    # config that predates the identity fix -- identity.py raises a clear
    # error if it needs one of these and it is not configured.
    age_col: str | None = None
    sex_col: str | None = None
    timestamp_col: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            data_path=raw["data_path"],
            sheet_name=raw["sheet_name"],
            patient_id_col=raw["patient_id_col"],
            legacy_mg_id_col=raw["legacy_mg_id_col"],
            instrument_col=raw["instrument_col"],
            target_col=raw["target_col"],
            target_map=dict(raw["target_map"]),
            wavelength_header_row_hint=raw["wavelength_header_row_hint"],
            wavelength_min_nm=float(raw["wavelength_min_nm"]),
            wavelength_max_nm=float(raw["wavelength_max_nm"]),
            random_seed=int(raw["random_seed"]),
            age_col=raw.get("age_col"),
            sex_col=raw.get("sex_col"),
            timestamp_col=raw.get("timestamp_col"),
        )

    def resolved_data_path(self) -> str:
        return self.data_path if os.path.isabs(self.data_path) else os.path.join(ROOT, self.data_path)


@dataclass(frozen=True)
class InstrumentConfig:
    id: str
    model: str
    serial_number: str
    raw_device_string: str
    has_both_classes: bool
    data: DataConfig

    @classmethod
    def from_yaml(cls, path: str) -> "InstrumentConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        instrument = raw["instrument"]
        data_config_path = raw["data_config"]
        if not os.path.isabs(data_config_path):
            data_config_path = os.path.join(ROOT, data_config_path)
        return cls(
            id=instrument["id"],
            model=instrument["model"],
            serial_number=instrument["serial_number"],
            raw_device_string=instrument["raw_device_string"],
            has_both_classes=bool(instrument["has_both_classes"]),
            data=DataConfig.from_yaml(data_config_path),
        )


class SchemaValidationError(ValueError):
    """Raised for violations that must stop the pipeline (e.g. missing patient_id)."""


@dataclass
class SchemaIssue:
    severity: str  # "FATAL" | "WARNING"
    code: str
    message: str
    n_affected: int = 0
    examples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "n_affected": self.n_affected,
            "examples": self.examples,
        }


def validate_schema(df, config: DataConfig) -> list[SchemaIssue]:
    """Checks the raw (post-load) table against the invariants required by
    build_project_and_exp.md section 2. Returns a list of issues; raises
    SchemaValidationError immediately only for conditions the pipeline cannot
    meaningfully continue past (missing patient_id column)."""
    issues: list[SchemaIssue] = []

    if config.patient_id_col not in df.columns:
        raise SchemaValidationError(
            f"patient_id column {config.patient_id_col!r} not found: cannot evaluate a model "
            "without a patient identifier (spectra of one patient could leak across train/test)."
        )
    if config.target_col not in df.columns:
        raise SchemaValidationError(f"target column {config.target_col!r} not found.")

    n_missing_pid = df[config.patient_id_col].isna().sum()
    if n_missing_pid:
        issues.append(SchemaIssue(
            "FATAL", "missing_patient_id",
            f"{n_missing_pid} rows have no {config.patient_id_col!r}; these rows cannot be used.",
            n_affected=int(n_missing_pid),
        ))

    dup_ids = df[config.patient_id_col][df.duplicated(subset=[config.patient_id_col], keep=False)]
    # duplicated GUIDs are expected (multiple spectra per visit) -- only flag if the
    # *combination* (patient_id, pulse index) repeats, i.e. an actual duplicated spectrum
    if "Pulses|Wavelenghts" in df.columns:
        spec_key_cols = [config.patient_id_col, "Pulses|Wavelenghts"]
        dup_spectra = df[df.duplicated(subset=spec_key_cols, keep=False)]
        if len(dup_spectra):
            issues.append(SchemaIssue(
                "WARNING", "duplicated_spectrum_id",
                f"{len(dup_spectra)} rows share both {config.patient_id_col!r} and pulse index "
                "with another row (possible duplicated spectrum).",
                n_affected=int(len(dup_spectra)),
            ))

    # one target value per patient_id
    mapped_target = df[config.target_col].map(config.target_map)
    n_unmapped = int((df[config.target_col].notna() & mapped_target.isna()).sum())
    if n_unmapped:
        issues.append(SchemaIssue(
            "WARNING", "unmapped_target_value",
            f"{n_unmapped} rows have a non-null {config.target_col!r} value outside target_map "
            f"{list(config.target_map)}.",
            n_affected=n_unmapped,
        ))
    target_per_patient = mapped_target.groupby(df[config.patient_id_col]).nunique(dropna=True)
    conflicting = target_per_patient[target_per_patient > 1]
    if len(conflicting):
        issues.append(SchemaIssue(
            "FATAL", "conflicting_target_per_patient",
            f"{len(conflicting)} patient_id values map to more than one target value.",
            n_affected=int(len(conflicting)),
            examples=conflicting.index.tolist()[:10],
        ))

    n_missing_instrument = df[config.instrument_col].isna().sum()
    if n_missing_instrument:
        issues.append(SchemaIssue(
            "WARNING", "missing_instrument",
            f"{n_missing_instrument} rows have no {config.instrument_col!r}.",
            n_affected=int(n_missing_instrument),
        ))

    return issues
