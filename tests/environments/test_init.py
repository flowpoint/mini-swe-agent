import pytest

from minisweagent.environments import get_environment_class
from minisweagent.environments.local import LocalEnvironment


class TestGetEnvironmentClass:
    def test_get_environment_class_local_shorthand(self):
        """Test that get_environment_class returns LocalEnvironment when given shorthand."""
        env_class = get_environment_class("local")
        assert env_class is LocalEnvironment

    def test_get_environment_class_docker_shorthand(self):
        """Docker belongs to the checkout-only benchmark runners."""
        with pytest.raises(ValueError, match="Unknown environment type: docker. Available: local"):
            get_environment_class("docker")

    def test_get_environment_class_invalid_spec(self):
        """Test that get_environment_class raises ValueError for invalid spec."""
        with pytest.raises(ValueError, match="Unknown environment type"):
            get_environment_class("invalid_environment")

    def test_get_environment_class_invalid_module(self):
        """Test that get_environment_class raises ValueError for non-existent module."""
        with pytest.raises(ValueError, match="Unknown environment type"):
            get_environment_class("nonexistent.module.Class")
