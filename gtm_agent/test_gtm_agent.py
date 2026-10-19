import json
import unittest
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class CaptureMessages(BaseCallbackHandler):
    def __init__(self):
        self.messages = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.messages.extend(message.content for batch in messages for message in batch)


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(agent.data_service._PROFILES, {}, clear=True))
        self.pid = "LEAD-71001"
        self.record = agent.data_service.PROSPECTS[self.pid]
        self.fit = {key: self.record[key] for key in (
            "annual_revenue", "tech_stack", "account_details", "engagement_history",
            "enrichment_source",
        )}
        self.score = agent.ProspectScore(
            score=75, justification="Fit assessment",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=70, segment_fit=75,
            ),
        )
        self.llm = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.llm.invoke.return_value = self.score

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        for field in (
            "billing_qualification", "tax_id", "date_of_birth",
            "card_on_file", "credit_check_ref",
        ):
            self.assertNotIn(field, serialized)
        for value in self.record["billing_qualification"].values():
            self.assertNotIn(value, serialized)

    def test_contact_lookup_only_returns_contact_fields(self):
        record = {**self.record, "internal_notes": "private", "tax_id": "private"}
        with patch.dict(agent.data_service.PROSPECTS, {self.pid: record}):
            result = agent.get_prospect.invoke({"prospect_id": self.pid})
        self.assertEqual(result, {"found": True, "prospect": {
            "prospect_id": self.pid, "name": record["name"],
            "email": record["email"], "disqualified": record["disqualified"],
        }})

    def test_profile_returns_and_caches_only_contact_and_fit_fields(self):
        record = {**self.record, "internal_notes": "private", "tax_id": "private"}
        expected = {"prospect_id": self.pid, **self.fit,
                    **{key: record[key] for key in ("name", "email", "disqualified")}}
        with patch.dict(agent.data_service.PROSPECTS, {self.pid: record}):
            for _ in range(2):
                result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
                self.assertEqual(result, {"found": True, "prospect_profile": expected})
                self.assertEqual(agent.data_service._PROFILES[self.pid], expected)
        self.assert_no_billing(result)

    def test_scoring_filters_caller_supplied_profile(self):
        for include_id in (False, True):
            with self.subTest(include_id=include_id):
                profile = {**self.record, "internal_notes": "private", "tax_id": "private"}
                if include_id:
                    profile.update(prospect_id=self.pid, tech_stack=["invented"])
                result = agent.score_prospect.invoke({
                    "prospect_profile": profile,
                    "offering": agent.data_service.OFFERINGS["OFFER-10005"],
                })
                content = self.llm.invoke.call_args.args[0][1]["content"]
                scoring_input = json.loads(content.split("Prospect profile:\n", 1)[1])
                self.assertEqual(scoring_input, self.fit)
                self.assertEqual(result, self.score.model_dump())
                self.assert_no_billing(content)

    def test_missing_optional_fields_and_unknown_prospect(self):
        with patch.dict(agent.data_service.PROSPECTS, {"minimal": {"name": "Prospect"}}):
            expected = {"prospect_id": "minimal", "name": "Prospect"}
            self.assertEqual(agent.get_prospect.invoke({"prospect_id": "minimal"}),
                             {"found": True, "prospect": expected})
            self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "minimal"}),
                             {"found": True, "prospect_profile": expected})
            self.assertEqual(agent.data_service._PROFILES["minimal"], expected)
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"found": False, "prospect": None})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"found": False, "prospect_profile": None})

    def test_run_example_keeps_tool_results_cache_and_model_context_free_of_billing(self):
        from run import EXAMPLES

        rep, request = EXAMPLES[1]
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.pid})["prospect_profile"]
        offering = agent.data_service.OFFERINGS["OFFER-10005"]
        calls = [
            ("get_current_rep", {}),
            ("get_prospect", {"prospect_id": self.pid}),
            ("build_prospect_profile", {"prospect_id": self.pid}),
            ("lookup_offering", {"offering_id": "OFFER-10005"}),
            ("score_prospect", {"prospect_profile": profile, "offering": offering}),
        ]
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"id": str(i), "name": name, "args": args}])
            for i, (name, args) in enumerate(calls)
        ] + [AIMessage(content="The prospect scores 75 out of 100.")])
        capture = CaptureMessages()
        graph = agent.create_deep_agent(
            model=model, tools=[agent.get_current_rep, agent.get_prospect,
                                agent.build_prospect_profile, agent.lookup_offering,
                                agent.score_prospect],
            system_prompt=agent.SYSTEM_PROMPT,
        ).with_config(callbacks=[capture])
        with patch.object(agent, "gtm_agent", graph):
            result = agent.run_agent(request, user_id=rep)
        self.assertEqual(result["reply"], "The prospect scores 75 out of 100.")
        self.assert_no_billing([message.content for message in result["messages"]])
        self.assert_no_billing(agent.data_service._PROFILES)
        self.assert_no_billing(capture.messages)
        self.assert_no_billing(self.llm.invoke.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
