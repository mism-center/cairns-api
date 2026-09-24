import os
import tempfile
from langfuse import Langfuse
from util.llm_helper import LLMFactory, DeferredLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from util.prompt_loader import get_prompt_text


class InputGuard:

    def __init__(self, config):
        self._instance = self._initialize(config)
        self.instance = self._instance

    def _initialize(self, config):
        if not getattr(config, "INPUT_GUARD_ENABLED", False):
            self.instance = RunnableLambda(lambda x: x)
            return self.instance

        # Imported lazily: nemoguardrails is only needed when the guard is
        # actually enabled, so the app can start without it otherwise.
        from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
        from nemoguardrails import RailsConfig

        temp_dir_root = config.TMP_DIR
        langfuse_client = None
        if config.LANGFUSE_ENABLED:
            langfuse_client = Langfuse(secret_key=config.LANGFUSE_SECRET_KEY,
                                       public_key=config.LANGFUSE_PUBLIC_KEY,
                                       host=config.LANGFUSE_HOST)
        rails_config = get_prompt_text("GUARDRAILS_CONFIG", config=config, langfuse_client=langfuse_client)
        rails_prompt = get_prompt_text("INPUT_GUARDRAILS", config=config, langfuse_client=langfuse_client)
        if not rails_config.strip() or not rails_prompt.strip():
            self.instance = RunnableLambda(lambda x: x)
            return self.instance
        rail_config_dir = InputGuard.setup_config_dir(temp_dir_root,
                                                      rails_config=rails_config,
                                                      rails_prompt=rails_prompt
                                                      )
        rails_config = RailsConfig.from_path(rail_config_dir)
        # use same model as generative model.
        def llm_factory():
            return LLMFactory.get_raw_llm(config)

        llm: BaseChatModel = DeferredLLM(llm_factory)
        self.instance = (RunnableRails(rails_config, llm))
        return self.instance

    @staticmethod
    def setup_config_dir(dir_root, rails_config, rails_prompt):
        temp_dir = tempfile.mkdtemp(dir=dir_root)
        guard_config_path = os.path.join(temp_dir, "config")
        os.makedirs(guard_config_path, exist_ok=True)
        config_file_path = os.path.join(guard_config_path, "config.yml")
        prompt_file_path = os.path.join(guard_config_path, "prompt.yml")
        with open(config_file_path, 'w') as config_file:
            config_file.write(rails_config)
        with open(prompt_file_path, 'w') as prompt_file:
            prompt_file.write(rails_prompt)
        return guard_config_path

    def __or__(self, other):
        return self._instance | other

    def __ror__(self, other):
        return other | self._instance

    def __getattr__(self, item):
        return getattr(self._instance, item)
