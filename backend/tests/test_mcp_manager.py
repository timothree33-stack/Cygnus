import asyncio
import pytest
from mcp.model_manifest import ModelManifest
from mcp.manager import Manager

@pytest.mark.asyncio
async def test_manifest_validate():
    m = ModelManifest(name="test", version="0.1", source_url="http://example.com/test")
    assert m.name == "test"
    m.validate()  # should not raise

@pytest.mark.asyncio
async def test_manager_download_and_launch(tmp_path):
    mgr = Manager(model_cache=str(tmp_path))
    await mgr.initialize()
    m = ModelManifest(name="tiny", version="0.0.1", source_url="http://example.com/tiny")
    path = await mgr.download_model(m)
    assert path is not None and (tmp_path / "tiny-0.0.1").exists()
    model_id = await mgr.launch_model(m)
    status = await mgr.get_model_status(model_id)
    assert status.get('status') == 'running'
