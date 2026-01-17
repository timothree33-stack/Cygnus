import pytest

from backend.agents.sample_agent import SampleAgent


@pytest.mark.asyncio
async def test_sample_agent_respond_returns_string():
    agent = SampleAgent("tester")
    res = await agent.respond(topic="unit-test", round=1)
    assert isinstance(res, str)
    assert "tester responds to 'unit-test' (round 1)" in res


@pytest.mark.asyncio
async def test_sample_agent_default_values():
    agent = SampleAgent()
    res = await agent.respond()
    assert "sample responds to" in res
