"""
SarvamAI LLM Client – OpenAI-compatible chat completions.
"""
import requests
import json
import logging
from config import SARVAM_API_KEY, SARVAM_BASE_URL, SARVAM_MODEL

logger = logging.getLogger(__name__)


class SarvamLLMClient:
    """Thin wrapper around Sarvam AI's chat completions endpoint."""

    def __init__(
        self,
        api_key: str = SARVAM_API_KEY,
        base_url: str = SARVAM_BASE_URL,
        model: str = SARVAM_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

    # ── core call ──────────────────────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,   # sarvam-105b uses tokens on reasoning; needs headroom
    ) -> str:
        """Send a chat-completion request and return the assistant reply."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.base_url}/v1/chat/completions"
        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=90)
            if not resp.ok:
                error_body = resp.text[:500]
                logger.error(
                    "Sarvam API %s error: %s\nURL: %s\nModel: %s\nBody: %s",
                    resp.status_code, resp.reason, url, self.model, error_body,
                )
                raise RuntimeError(
                    f"Sarvam API {resp.status_code} {resp.reason}: {error_body}"
                )
            data = resp.json()
            msg = data["choices"][0]["message"]
            # sarvam-105b may return null content when all tokens go to reasoning;
            # fall back to reasoning_content in that case
            content = msg.get("content") or msg.get("reasoning_content") or ""
            if not content:
                logger.warning("Sarvam API returned empty content. Full response: %s", data)
            return content
        except requests.exceptions.HTTPError as e:
            logger.error("Sarvam API HTTP error: %s – %s", e, resp.text)
            raise
        except Exception as e:
            logger.error("Sarvam API error: %s", e)
            raise

    # ── convenience helpers ────────────────────────────────────────────────
    def generate_answer(self, question: str, context_passages: list[str], temperature: float = 0.3) -> str:
        """Generate an evidence-based medical answer from retrieved passages."""
        context_block = "\n\n---\n\n".join(
            f"[Passage {i+1}]\n{p}" for i, p in enumerate(context_passages)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a biomedical evidence synthesis assistant. "
                    "Answer the user's medical question using ONLY the provided evidence passages. "
                    "Structure your answer as a set of numbered factual claims. "
                    "Each claim must be directly supported by the evidence. "
                    "If the evidence is insufficient, say 'Insufficient evidence to answer this question.' "
                    "Do NOT hallucinate or add information beyond the evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Question\n{question}\n\n"
                    f"## Evidence Passages\n{context_block}\n\n"
                    "Provide your evidence-based answer as numbered claims:"
                ),
            },
        ]
        return self.chat(messages, temperature=temperature)

    def extract_claims(self, answer_text: str) -> list[str]:
        """Extract individual factual claims from a generated answer."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise claim extractor. Given an answer text, "
                    "extract every individual factual claim as a separate item. "
                    "Return ONLY a JSON array of strings, each being one atomic claim. "
                    "Do not add any commentary. Example: "
                    '[\"Aspirin reduces inflammation.\", \"The dosage is 81mg daily.\"]'
                ),
            },
            {
                "role": "user",
                "content": f"Extract claims from:\n\n{answer_text}",
            },
        ]
        raw = self.chat(messages, temperature=0.1, max_tokens=1024)
        # Parse the JSON array from the response
        try:
            # Find JSON array in response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
        # Fallback: split by newlines and clean up
        claims = []
        for line in raw.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ").strip()
            if line and len(line) > 10:
                claims.append(line)
        return claims if claims else [answer_text]

    def rewrite_answer(
        self, question: str, original_answer: str,
        supported_claims: list[str], unsupported_claims: list[str],
        context_passages: list[str], temperature: float = 0.3,
    ) -> str:
        """Rewrite the answer, removing unsupported claims."""
        context_block = "\n\n---\n\n".join(
            f"[Passage {i+1}]\n{p}" for i, p in enumerate(context_passages)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a biomedical evidence synthesis assistant performing self-correction. "
                    "You will receive an original answer, a list of SUPPORTED claims, and a list of "
                    "UNSUPPORTED claims that failed NLI verification. "
                    "Rewrite the answer keeping ONLY the supported claims. "
                    "Remove or correct any unsupported claims. "
                    "If no claims are supported, respond with: "
                    "'Insufficient evidence to answer this question reliably.' "
                    "Structure the rewritten answer as numbered claims."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Question\n{question}\n\n"
                    f"## Evidence Passages\n{context_block}\n\n"
                    f"## Original Answer\n{original_answer}\n\n"
                    f"## Supported Claims\n" +
                    "\n".join(f"✓ {c}" for c in supported_claims) +
                    f"\n\n## Unsupported Claims (REMOVE THESE)\n" +
                    "\n".join(f"✗ {c}" for c in unsupported_claims) +
                    "\n\nRewrite the corrected answer:"
                ),
            },
        ]
        return self.chat(messages, temperature=temperature)
