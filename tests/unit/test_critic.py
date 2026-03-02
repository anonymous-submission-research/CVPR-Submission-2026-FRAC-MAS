import unittest
from unittest.mock import MagicMock, patch
from src.medai.agents.critic_agent import CriticAgent
from src.medai.utils.consensus import evaluate_consensus

class TestCriticAgent(unittest.TestCase):
    
    @patch('src.medai.agents.critic_agent.MedGemmaClient')
    def test_review_diagnosis_yes(self, MockClient):
        # Setup
        mock_instance = MockClient.return_value
        mock_instance.predict.return_value = "Yes, the fracture is perpendicular to the shaft."
        
        agent = CriticAgent(mode="local")
        image_mock = MagicMock()
        
        # Action
        result = agent.review_diagnosis(
            image=image_mock,
            prediction_label="Transverse Fracture",
            prediction_confidence=0.85,
            context_definition="Break perpendicular to the bone."
        )
        
        # Assert
        self.assertEqual(result["verdict"], "yes")
        self.assertFalse(result["flagged_for_human"])
        self.assertIn("Yes", result["explanation"])

    @patch('src.medai.agents.critic_agent.MedGemmaClient')
    def test_review_diagnosis_no(self, MockClient):
        # Setup
        mock_instance = MockClient.return_value
        mock_instance.predict.return_value = "No, this looks like an oblique fracture."
        
        agent = CriticAgent(mode="local")
        image_mock = MagicMock()
        
        # Action
        result = agent.review_diagnosis(
            image=image_mock,
            prediction_label="Transverse Fracture",
            prediction_confidence=0.85,
            context_definition="Break perpendicular to the bone."
        )
        
        # Assert
        self.assertEqual(result["verdict"], "no")
        self.assertTrue(result["flagged_for_human"])


class TestConsensus(unittest.TestCase):
    
    def test_consensus_agreement(self):
        vision = {"label": "A", "confidence": 0.9}
        critic = {"verdict": "yes", "critic_confidence": 0.8} # Score ~0.8
        # Delta 0.1 <= 0.2
        
        result = evaluate_consensus(vision, critic, delta_threshold=0.2)
        self.assertEqual(result["final_decision"], "approved")
        self.assertFalse(result["human_review_required"])

    def test_consensus_contradiction(self):
        vision = {"label": "A", "confidence": 0.9}
        critic = {"verdict": "no", "critic_confidence": 0.8} # Score -> 1 - 0.8 = 0.2
        # Delta |0.9 - 0.2| = 0.7 > 0.2
        
        result = evaluate_consensus(vision, critic, delta_threshold=0.2)
        self.assertEqual(result["final_decision"], "flagged")
        self.assertTrue(result["human_review_required"])

    def test_high_uncertainty_flag(self):
        vision = {"label": "A", "confidence": 0.9}
        critic = {"verdict": "uncertain", "critic_confidence": 0.0} # Fallback score 0.5
        # Delta |0.9 - 0.5| = 0.4 > 0.2
        
        result = evaluate_consensus(vision, critic, delta_threshold=0.2)
        self.assertEqual(result["final_decision"], "flagged")

if __name__ == '__main__':
    unittest.main()
