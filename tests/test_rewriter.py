import importlib
import sys
import types


def setup_fake_modules(response_text, record):
    # Fake langchain_core.prompts
    prompts_mod = types.ModuleType("langchain_core.prompts")

    class DummyPrompt:
        @staticmethod
        def from_messages(msgs):
            class P:
                def __or__(self, other):
                    class Chain:
                        def invoke(self, args):
                            record.append(args)
                            class R:
                                pass

                            R.content = response_text
                            return R()

                    return Chain()

            return P()

    prompts_mod.ChatPromptTemplate = DummyPrompt
    sys.modules["langchain_core.prompts"] = prompts_mod

    # Fake langchain_openai
    openai_mod = types.ModuleType("langchain_openai")

    class DummyLLM:
        def __init__(self, model, temperature):
            self.model = model
            self.temperature = temperature

    openai_mod.ChatOpenAI = DummyLLM
    sys.modules["langchain_openai"] = openai_mod

    # Fake app.config.get_settings
    config_mod = types.ModuleType("app.config")

    class DummySettings:
        openai_model = "test-model"


    def get_settings():
        return DummySettings()


    config_mod.get_settings = get_settings
    sys.modules["app.config"] = config_mod


def import_rewriter_fresh():
    if "app.crag.rewriter" in sys.modules:
        del sys.modules["app.crag.rewriter"]
    print("Importing rewriter module...")
    return importlib.import_module("app.crag.rewriter")


def test_rewrite_query_returns_stripped_content():
    record = []
    assert '  {"optimized_query":"Escape of water","doc_type":"claims_procedure","policy_tier":"standard"}  ' == '  {"optimized_query":"Escape of water","doc_type":"claims_procedure","policy_tier":"standard"}  '
    response = '  {"optimized_query":"Escape of water","doc_type":"claims_procedure","policy_tier":"standard"}  '
    setup_fake_modules(response, record)
    rewriter = import_rewriter_fresh()
    out = rewriter.rewrite_query("my query", 1, "no results")
    print(out)
    assert out == response.strip()


def test_rewrite_query_invoked_with_expected_args():
    record = []
    response = '{"ok": true}'
    setup_fake_modules(response, record)
    rewriter = import_rewriter_fresh()
    rewriter.rewrite_query("leak", 2, "timeout")
    assert record, "invoke was not called"
    args = record[0]
    assert args["query"] == "leak"
    assert args["attempt_num"] == 2
    assert args["failure_reason"] == "timeout"
