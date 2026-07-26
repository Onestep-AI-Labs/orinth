from dataclasses import dataclass, field

from app.schemas import AdvancedParameterSpec, TrainingModelOption


@dataclass(frozen=True)
class TrainingModelDefinition:
    id: str
    name: str
    family: str
    task_types: list[str]
    source: str
    runnable: bool
    needs_download: bool
    description: str
    defaults: dict = field(default_factory=dict)
    # Catalog-declared advanced hyperparameters. Empty by default so untouched
    # catalogs serialize exactly as before.
    advanced_parameters: list[AdvancedParameterSpec] = field(default_factory=list)

    def to_option(self) -> TrainingModelOption:
        return TrainingModelOption(
            id=self.id,
            name=self.name,
            family=self.family,
            task_types=self.task_types,  # type: ignore[arg-type]
            source=self.source,  # type: ignore[arg-type]
            runnable=self.runnable,
            needs_download=self.needs_download,
            description=self.description,
            defaults=dict(self.defaults),
            advanced_parameters=list(self.advanced_parameters),
        )
