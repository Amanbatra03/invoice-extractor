import os


def test_api_dockerfile_exists():
    assert os.path.exists("Dockerfile.api")


def test_frontend_dockerfile_exists():
    assert os.path.exists("Dockerfile.frontend")


def test_docker_compose_exists():
    assert os.path.exists("docker-compose.yml")
