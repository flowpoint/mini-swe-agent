import pytest

from minisweagent.environments import get_environment_class
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.local import LocalEnvironment


class TestGetEnvironmentClass:
    def test_get_environment_class_local_shorthand(self):
        """Test that get_environment_class returns LocalEnvironment when given shorthand."""
        env_class = get_environment_class("local")
        assert env_class is LocalEnvironment

    def test_get_environment_class_docker_shorthand(self):
        """Docker is the only optional environment supported by the registry."""
        env_class = get_environment_class("docker")
        assert env_class is DockerEnvironment

    def test_get_environment_class_invalid_spec(self):
        """Test that get_environment_class raises ValueError for invalid spec."""
        with pytest.raises(ValueError, match="Unknown environment type"):
            get_environment_class("invalid_environment")

    def test_get_environment_class_invalid_module(self):
        """Test that get_environment_class raises ValueError for non-existent module."""
        with pytest.raises(ValueError, match="Unknown environment type"):
            get_environment_class("nonexistent.module.Class")
