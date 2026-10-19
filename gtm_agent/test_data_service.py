import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = deepcopy(data_service.PROSPECTS)
        self.profiles = deepcopy(data_service._PROFILES)
        self.addCleanup(patch.stopall)
        patch.object(data_service, "PROSPECTS", self.records).start()
        patch.object(data_service, "_PROFILES", self.profiles).start()
        self.prospect_id = "LEAD-71001"

    def test_new_technology_persists_and_rebuilds_cached_profile(self):
        before = deepcopy(self.records)
        cached = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Kafka", cached["prospect_profile"]["tech_stack"])
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")
        expected = before[self.prospect_id]["tech_stack"] + ["Kafka"]
        self.assertEqual(result, {"updated": True, "found": True, "tech_stack": expected})
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), expected)
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(rebuilt["prospect_profile"]["tech_stack"], expected)
        before[self.prospect_id]["tech_stack"] = expected
        self.assertEqual(self.records, before)

    def test_existing_technology_preserves_contract_without_duplicates(self):
        data_service.update_prospect_info(self.prospect_id, "Kafka")
        before = deepcopy(self.records)
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")
        self.assertEqual(result, {
            "updated": True, "found": True,
            "tech_stack": before[self.prospect_id]["tech_stack"],
        })
        self.assertEqual(result["tech_stack"].count("Kafka"), 1)
        self.assertEqual(self.records, before)

    def test_unknown_prospect_does_not_mutate_records_or_profiles(self):
        gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        records = deepcopy(self.records)
        profiles = deepcopy(self.profiles)
        self.assertEqual(data_service.update_prospect_info("unknown", "Kafka"), {
            "updated": False, "found": False,
        })
        self.assertEqual(self.records, records)
        self.assertEqual(self.profiles, profiles)

    def test_add_kafka_then_score_uses_saved_stack_even_with_old_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = gtm_agent.lookup_offering.invoke({"offering_id": "OFFER-10005"})
        updated = gtm_agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Kafka",
        })
        self.assertTrue(updated["updated"])
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"],
                "offering": offering["offering"],
            })
        messages = scoring_llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertIn("Kafka", scored_profile["tech_stack"])
        self.assertEqual(scored_profile["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))


if __name__ == "__main__":
    unittest.main()
