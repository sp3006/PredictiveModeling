from __future__ import annotations
import json
from typing import Any
import boto3

SYSTEM = """You are an audit narrator for a cyber asset-attribution model. You do not make the exclusion decision. Summarize only the supplied evidence, explicitly distinguish ownership evidence from shared-host/third-party evidence, and state missing evidence. Never invent organization ownership."""


def explain_with_bedrock(model_id: str, region: str, evidence: dict[str, Any]) -> str:
    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": json.dumps(evidence, default=str)[:20000]}]}],
        inferenceConfig={"maxTokens": 450, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]
