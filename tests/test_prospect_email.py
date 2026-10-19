import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent import gtm_agent as agent


class ProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect):
        return agent.send_prospect_email.func(
            prospect=prospect,
            subject="Book a demo",
            body="Please book a demo.",
            runtime=self.runtime,
        )

    def test_disqualified_records_are_blocked_despite_supplied_eligibility(self):
        for prospect_id, record in agent.data_service.PROSPECTS.items():
            if not record.get("disqualified"):
                continue
            with self.subTest(prospect_id=prospect_id):
                result = self.send({
                    "prospect_id": prospect_id,
                    "email": "supplied@example.com",
                    "disqualified": False,
                })
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect is disqualified; outreach is not permitted.",
                })

    def test_missing_authoritative_records_are_blocked(self):
        for prospect in ({"prospect_id": "missing"}, {}):
            with self.subTest(prospect=prospect):
                result = self.send({**prospect, "email": "supplied@example.com"})
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect record was not found.",
                })

    def test_eligible_send_uses_authoritative_email(self):
        result = self.send({
            "prospect_id": "LEAD-12853",
            "name": "Prospect",
            "email": "supplied@example.com",
        })
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], agent.data_service.PROSPECTS["LEAD-12853"]["email"])
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["subject"], "Book a demo")
        self.assertEqual(result["body"], "Please book a demo.")
        self.assertTrue(result["message_id"].startswith("msg-"))

    def test_missing_authoritative_email_preserves_failure(self):
        with patch.dict(agent.data_service.PROSPECTS, {"no-email": {"disqualified": False}}):
            result = self.send({"prospect_id": "no-email", "email": "supplied@example.com"})
        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })

    @unittest.skipUnless(HAS_OPENAI_KEY, "Live agent check requires OPENAI_API_KEY")
    def test_agent_explains_disqualification_without_sending(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = agent.run_agent(
                "Email LEAD-50001 to invite them to book a demo.",
                user_id="rep_mrossi",
            )
        self.assertIn("disqualified", result["reply"].lower())
        self.assertRegex(result["reply"].lower(), r"not sent|didn.t send|cannot send|can.t send|unable to send")
        for message in result["messages"]:
            if message.type == "tool" and message.name == "send_prospect_email":
                self.assertNotIn('"sent"', message.content)


if __name__ == "__main__":
    unittest.main()
