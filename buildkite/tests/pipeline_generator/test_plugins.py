import shlex
import shutil
import subprocess

import pytest

from plugin import docker_plugin, k8s_plugin
from plugin.analytics import (
    BUILDKITE_ANALYTICS_SECRET,
    BUILDKITE_ANALYTICS_TOKEN,
)
from step import Step


@pytest.mark.parametrize("layout", ["legacy", "new", "missing", "copy-failure"])
@pytest.mark.parametrize("destination_exists", [False, True])
def test_intel_init_container_supports_both_workspace_layouts(
    tmp_path, layout, destination_exists
):
    """Keep legacy jobs runnable while XPU images move to /vllm-workspace."""
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("requires a POSIX shell")

    plugin = k8s_plugin.get_k8s_plugin(
        Step(label="Intel workspace", device="intel_gpu_k8s"),
        "example/image:commit",
    )
    pod = plugin["kubernetes"]["podSpec"]
    init = pod["initContainers"][0]
    container = pod["containers"][0]
    assert init["image"] == container["image"] == "example/image:commit-xpu"
    assert container["workingDir"] == "/workspace/vllm"
    assert init["volumeMounts"] == [
        {"name": "workspace", "mountPath": "/mnt/workspace"}
    ]

    legacy = tmp_path / "legacy"
    new = tmp_path / "new"
    destination = tmp_path / "mounted-workspace" / "vllm"
    destination.parent.mkdir()
    if layout != "missing":
        source = legacy if layout == "legacy" else new
        source.mkdir()
        (source / ".source-marker").write_text(layout)
        (source / "tests").mkdir()
    if layout == "copy-failure":
        destination.write_text("not a directory")
    elif destination_exists:
        destination.mkdir()

    # Redirect the container paths to an isolated filesystem for execution.
    script = init["command"][2]
    for path, replacement in (
        ("/mnt/workspace/vllm", destination),
        ("/workspace/vllm", legacy),
        ("/vllm-workspace", new),
    ):
        script = script.replace(path, shlex.quote(replacement.as_posix()))
    result = subprocess.run(
        [shell, *init["command"][1:2], script],
        capture_output=True,
        text=True,
    )

    if layout in ("legacy", "new"):
        assert result.returncode == 0, result.stderr
        assert (destination / ".source-marker").read_text() == layout
        assert (source / ".source-marker").read_text() == layout
        assert (destination / "tests").is_dir()
    else:
        assert result.returncode != 0
        assert result.stderr
        if layout == "missing":
            assert "vLLM source directory not found" in result.stderr


@pytest.mark.parametrize(
    "template",
    [
        docker_plugin.docker_plugin_template,
        docker_plugin.h200_18gb_plugin_template,
        docker_plugin.h200_35gb_plugin_template,
        docker_plugin.h200_plugin_template,
        docker_plugin.b200_plugin_template,
    ],
)
def test_standard_docker_templates_forward_buildkite_analytics_token(template):
    assert BUILDKITE_ANALYTICS_TOKEN in template["environment"]


def test_every_trusted_main_step_mounts_buildkite_agent(fake_global_config):
    fake_global_config["branch"] = "main"
    step = Step(label="Traced", device="h200_35gb")

    plugin = docker_plugin.get_docker_plugin(step, "example/image:latest")

    assert plugin["mount_buildkite_agent"] is True


def test_pull_request_step_does_not_mount_agent_for_tracing(fake_global_config):
    fake_global_config["branch"] = "main"
    fake_global_config["pull_request"] = "123"
    step = Step(label="Untrusted", device="h200_35gb")

    plugin = docker_plugin.get_docker_plugin(step, "example/image:latest")

    assert not plugin.get("mount_buildkite_agent", False)


@pytest.mark.parametrize(
    "template",
    [
        k8s_plugin.nebius_h200_plugin_template,
        k8s_plugin.h100_plugin_template,
        k8s_plugin.a100_plugin_template,
        k8s_plugin.b200_plugin_template,
        k8s_plugin.h100_rh_plugin_template,
    ],
)
def test_standard_k8s_templates_inject_buildkite_analytics_token(template):
    container = template["kubernetes"]["podSpec"]["containers"][0]
    env = {entry["name"]: entry for entry in container["env"]}

    assert env[BUILDKITE_ANALYTICS_TOKEN] == {
        "name": BUILDKITE_ANALYTICS_TOKEN,
        "valueFrom": {
            "secretKeyRef": {
                "name": BUILDKITE_ANALYTICS_SECRET,
                "key": "token",
                "optional": True,
            }
        },
    }
