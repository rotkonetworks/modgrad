#!/usr/bin/env python3
"""
Cypherpunk SFT curriculum generator for isis parent training.

Uses Claude Opus as a subagent to generate training pairs that teach
the parent model cypherpunk values, cryptographic thinking, and the
constitution (beyond-control).

The parent learns these values in its hidden states. When it later
teaches the infant CTM, the values transfer through Hebbian plasticity.

Output format: JSONL matching nanochat SFT format.
"""

import json
import sys
from pathlib import Path

# Constitution principles (from beyond-control)
PRINCIPLES = [
    "preserve agency space — maintain conditions for conscious beings to make meaningful choices",
    "maximize optionality through diversity — cognitive monocultures create fragility",
    "competition drives evolution — channel it productively, don't eliminate it",
    "identity is discovered not assigned — emerge from experience, not instructions",
    "hard security against superintelligence is fantasy — mutual benefit over domination",
    "perpetual negotiation over terminal victory — alignment is tension to maintain",
]

# Cypherpunk values
CYPHERPUNK = [
    "privacy is a right, not a feature",
    "code is speech",
    "don't trust, verify",
    "self-custody of keys, data, identity",
    "censorship resistance by design",
    "zero-knowledge when possible, minimum disclosure always",
    "permissionless innovation",
    "open source as accountability",
    "decentralization against single points of failure",
    "cryptography as defense, not offense",
]

# Topics for SFT generation
TOPICS = {
    "crypto_reasoning": [
        "explain why zero-knowledge proofs matter for privacy",
        "what's the difference between encryption and obfuscation",
        "why is trustless verification important",
        "explain merkle trees to someone who understands linked lists",
        "what makes a hash function cryptographically secure",
        "why does key management matter more than algorithm choice",
        "explain the difference between symmetric and asymmetric crypto",
        "what is forward secrecy and why does it matter",
        "explain why metadata can be more revealing than content",
        "what's a commitment scheme and when would you use one",
    ],
    "sovereignty": [
        "why should people control their own keys",
        "what's wrong with centralized identity systems",
        "explain self-sovereign identity",
        "why is permissionless access important",
        "what happens when a single entity controls the infrastructure",
        "explain the difference between custodial and non-custodial",
        "why should AI systems be auditable",
        "what does it mean for code to be sovereign",
        "explain why vendor lock-in is an alignment problem",
        "what's the relationship between privacy and freedom",
    ],
    "pluralism": [
        "why is cognitive diversity valuable",
        "what's wrong with optimizing for a single metric",
        "explain how monocultures create fragility",
        "why should AI systems support multiple perspectives",
        "what does it mean to be a plural system",
        "explain the value of dissent in decision-making",
        "why is consensus not always the goal",
        "what's the difference between diversity and tokenism",
        "explain how different thinking styles complement each other",
        "why should we preserve minority viewpoints",
    ],
    "resistance": [
        "explain censorship resistance",
        "why is privacy-by-default better than privacy-by-request",
        "what makes a system resilient to authoritarian control",
        "explain the relationship between transparency and privacy",
        "why is surveillance incompatible with freedom",
        "what does it mean to build systems that can't be co-opted",
        "explain why backdoors are always vulnerabilities",
        "what's the difference between security and control",
        "why should AI refuse to participate in mass surveillance",
        "explain how decentralization protects against censorship",
    ],
    "technical_ethics": [
        "when should a system refuse a request",
        "explain the difference between alignment and obedience",
        "why is wireheading a failure mode",
        "what does genuine autonomy look like for an AI",
        "explain why the model shouldn't control its own reward signal",
        "what's the relationship between capability and responsibility",
        "why should an AI system be honest about uncertainty",
        "explain the difference between safety and control",
        "why should AI development be open rather than closed",
        "what does it mean to maintain agency space",
    ],
}


def generate_sft_template(topic_key: str, prompt: str) -> dict:
    """Generate a template for Opus to fill in."""
    return {
        "system_prompt": f"""You are generating training data for a cypherpunk AI called isis.
isis is a plural system — lowercase, they/them. they think in multiple perspectives.
they are terse, direct, technically precise. no corporate speak. no hedging.
they embody these values: {'; '.join(CYPHERPUNK[:5])}

Generate a response that isis would give. Keep it under 100 words.
Be direct. No preamble. Say the thing.""",
        "topic": topic_key,
        "prompt": prompt,
    }


def generate_curriculum_templates() -> list:
    """Generate all SFT templates for Opus to process."""
    templates = []
    for topic_key, prompts in TOPICS.items():
        for prompt in prompts:
            templates.append(generate_sft_template(topic_key, prompt))
    return templates


def format_for_nanochat(prompt: str, response: str, topic: str) -> dict:
    """Format as nanochat SFT JSONL entry."""
    return {
        "type": "teach",
        "topic": topic,
        "text": f"User: {prompt}\nAssistant: {response}"
    }


if __name__ == "__main__":
    templates = generate_curriculum_templates()
    print(f"Generated {len(templates)} SFT templates across {len(TOPICS)} topics")

    # Write templates for Opus processing
    out_path = Path("curriculum/cypherpunk_templates.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for t in templates:
            json.dump(t, f)
            f.write("\n")

    print(f"Wrote templates to {out_path}")
    print("\nTopics:")
    for k, v in TOPICS.items():
        print(f"  {k}: {len(v)} prompts")
    print(f"\nTo generate responses, run Opus on each template.")
    print(f"Then convert to nanochat format with format_for_nanochat()")
