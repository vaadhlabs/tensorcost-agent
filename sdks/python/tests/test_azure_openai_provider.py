"""An Azure OpenAI client reports its calls as azure_openai, not openai.

Before, openai.AzureOpenAI (an openai.OpenAI subclass) was reported as
"openai": Azure spend landed under OpenAI, a budget scoped to azure_openai
never saw it, and it couldn't be paired with the Azure bill line.
"""
from tensorcost._applied_mode import openai_provider_for


class _Client:
    def __init__(self, base_url=None):
        self.base_url = base_url


class AzureOpenAI(_Client):
    pass


def test_azure_hosts_and_class_are_azure_openai():
    assert openai_provider_for(_Client("https://tc-zelvora-aoai-06a901.openai.azure.com/openai/")) == "azure_openai"
    assert openai_provider_for(_Client("https://acme.cognitiveservices.azure.com/openai")) == "azure_openai"
    assert openai_provider_for(_Client("https://acme.services.ai.azure.com/openai/v1")) == "azure_openai"
    assert openai_provider_for(AzureOpenAI("https://gateway.example.com/openai")) == "azure_openai"


def test_openai_and_compatible_endpoints_stay_openai():
    assert openai_provider_for(_Client("https://api.openai.com/v1")) == "openai"
    assert openai_provider_for(_Client("https://openrouter.ai/api/v1")) == "openai"
    assert openai_provider_for(_Client()) == "openai"
    assert openai_provider_for(None) == "openai"
