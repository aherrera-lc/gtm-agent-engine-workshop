import importlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langsmith import Client, tracing_context

from gtm_agent import data_service


ROOT = Path(__file__).resolve().parent.parent


class TracingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LANGSMITH_TRACING": "false",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("LANGSMITH_REVISION_ID", None)

    def startup(self, dotenv_text=""):
        with patch("dotenv.load_dotenv", side_effect=lambda **kwargs: load_dotenv(
            stream=io.StringIO(dotenv_text), **kwargs
        )), patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
            module = importlib.import_module("gtm_agent.gtm_agent")
            return importlib.reload(module)

    def test_detects_application_checkout_from_other_working_directory(self):
        expected = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            subprocess.run(["git", "init", "--quiet", directory], check=True)
            try:
                os.chdir(directory)
                self.startup()
            finally:
                os.chdir(previous)
        self.assertEqual(os.environ["LANGSMITH_REVISION_ID"], expected)

    def test_caller_revision_survives_dotenv_without_git(self):
        os.environ["LANGSMITH_REVISION_ID"] = "caller-release"
        with patch("subprocess.run", side_effect=AssertionError("git invoked")):
            self.startup("LANGSMITH_REVISION_ID=configured-release")
        self.assertEqual(os.environ["LANGSMITH_REVISION_ID"], "caller-release")

    def test_configured_revision_without_git(self):
        with patch("subprocess.run", side_effect=AssertionError("git invoked")):
            self.startup("LANGSMITH_REVISION_ID=configured-release")
        self.assertEqual(os.environ["LANGSMITH_REVISION_ID"], "configured-release")

    def test_git_failure_or_empty_output_leaves_revision_unset(self):
        for result in (FileNotFoundError(), subprocess.CalledProcessError(128, "git"),
                       subprocess.CompletedProcess([], 0, stdout="\n")):
            with self.subTest(result=result):
                os.environ.pop("LANGSMITH_REVISION_ID", None)
                kwargs = {"side_effect": result} if isinstance(result, Exception) else {
                    "return_value": result
                }
                with patch("subprocess.run", **kwargs):
                    self.startup("LANGSMITH_REVISION_ID=   ")
                self.assertNotIn("LANGSMITH_REVISION_ID", os.environ)

    def test_data_access_emits_named_tool_spans(self):
        for name, args in (
            ("get_offering", ("offering",)),
            ("get_prospect_record", ("prospect",)),
            ("get_rep", ("rep",)),
            ("update_prospect_info", ("prospect", "Python")),
        ):
            with self.subTest(name=name):
                spans = []
                with tracing_context(enabled="local"), patch.dict(
                    data_service.PROSPECTS, {"prospect": {"tech_stack": []}}
                ):
                    getattr(data_service, name)(*args, langsmith_extra={"on_end": spans.append})
                self.assertEqual(len(spans), 1)
                self.assertEqual((spans[0].name, spans[0].run_type), (name, "tool"))
                self.assertIsNone(spans[0].error)

    def test_root_trace_contains_checkout_revision_and_data_access_children(self):
        module = self.startup()
        expected = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        client = Client(api_key="test-key", auto_batch_tracing=False)
        self.addCleanup(client.close)

        def respond(inputs):
            data_service.get_offering("missing")
            data_service.get_prospect_record("missing")
            data_service.get_rep("missing")
            data_service.update_prospect_info("missing", "Python")
            return {"messages": [AIMessage(content="done")]}

        with patch.object(module, "gtm_agent", RunnableLambda(respond)), patch.object(
            Client, "request_with_retries", return_value=Mock()
        ) as request, tracing_context(enabled=True, client=client):
            result = module.run_agent("lookup offering", user_id="rep")
        runs = [json.loads(call.kwargs["request_kwargs"]["data"])
                for call in request.call_args_list if call.args[0] == "POST"]
        root = next(run for run in runs if str(run["id"]) == result["run_id"])
        self.assertEqual(root["extra"]["metadata"]["revision_id"], expected)
        children = {run["name"] for run in runs
                    if run.get("parent_run_id") == root["id"] and run["run_type"] == "tool"}
        self.assertEqual(children, {"get_offering", "get_prospect_record", "get_rep",
                                    "update_prospect_info"})

    def test_env_example_documents_optional_revision(self):
        example = (ROOT / ".env.example").read_text()
        self.assertIn("# LANGSMITH_REVISION_ID=", example)
        self.assertNotIn("LANGSMITH_REVISION_ID", {
            line.split("=", 1)[0] for line in example.splitlines()
            if line and not line.startswith("#")
        })


if __name__ == "__main__":
    unittest.main()
