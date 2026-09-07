# -*- coding: utf-8 -*-
"""LLM 客户端：智谱 GLM-4-Flash（OpenAI 兼容协议）。"""
from __future__ import annotations

import os

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
ZHIPU_MODEL = "glm-4-flash"


def has_api_key() -> bool:
    return bool(os.environ.get("ZHIPU_API_KEY"))


def get_llm():
    """ChatOpenAI 兼容客户端（温度 0 保证规划输出稳定）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=ZHIPU_BASE_URL,
        model=ZHIPU_MODEL,
        api_key=os.environ.get("ZHIPU_API_KEY", "missing"),
        temperature=0,
        timeout=60,
        max_retries=2,
    )
