"""Custom exceptions for the ai-text-toolkit."""


class AITextToolkitError(Exception):
    """Base exception for all ai-text-toolkit errors."""
    pass


class DetectorError(AITextToolkitError):
    """Raised when detector operations fail."""
    pass


class DetectorNotReadyError(DetectorError):
    """Raised when detector is used before loading a trained model."""
    pass


class DetectorLoadError(DetectorError):
    """Raised when classifier model fails to load."""
    pass


class FeatureExtractionError(DetectorError):
    """Raised when feature extraction fails."""
    pass


class PerplexityError(DetectorError):
    """Raised when perplexity computation fails."""
    pass


class HumanizerError(AITextToolkitError):
    """Raised when humanizer operations fail."""
    pass


class HumanizerNotConfiguredError(HumanizerError):
    """Raised when humanizer is used without a configured backend."""
    pass


class BackendError(HumanizerError):
    """Raised when a generation backend fails."""
    pass


class BackendNotAvailableError(BackendError):
    """Raised when a backend is not available (e.g., missing dependencies)."""
    pass


class ModelNotFoundError(BackendError):
    """Raised when a model file is not found."""
    pass


class ExternalProviderError(HumanizerError):
    """Raised when an external provider API call fails."""
    pass


class ExternalProviderNotAvailableError(ExternalProviderError):
    """Raised when an external provider is not available or misconfigured."""
    pass


class ConfigurationError(AITextToolkitError):
    """Raised when configuration is invalid or missing."""
    pass


class TrainingError(AITextToolkitError):
    """Raised when training fails."""
    pass


class DataLoadError(TrainingError):
    """Raised when training data cannot be loaded."""
    pass