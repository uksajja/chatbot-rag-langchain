from imrag.core.registries import LLMS


@LLMS.register("openai")
def openai_llm(model: str = "gpt-4o-mini", **kw):
    """Create an OpenAI chat model.

    The API key is read from the OPENAI_API_KEY environment variable.

    Args:
        model: OpenAI model name.
        **kw: Passed through to ChatOpenAI. Common options:
            temperature, max_tokens, max_retries, timeout.

    YAML example:
        llm:
          type: openai
          params:
            model: gpt-4o-mini
            temperature: 0.1
            max_tokens: 800
            max_retries: 3
            timeout: 30
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, **kw)


@LLMS.register("anthropic")
def anthropic_llm(model: str, **kw):
    """Create an Anthropic chat model.

    The API key is read from the ANTHROPIC_API_KEY environment variable.

    Args:
        model: Anthropic model id (no default, so the choice is always explicit).
        **kw: Passed through to ChatAnthropic. Common options:
            temperature, max_tokens.

    YAML example:
        llm:
          type: anthropic
          params:
            model: claude-sonnet-5-5
            temperature: 0.1
            max_tokens: 800
    """
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, **kw)


@LLMS.register("google_genai")
def gemini_llm(model: str, **kw):
    """Create a Google Gemini chat model.

    The API key is read from the GOOGLE_API_KEY environment variable.
    Requires the langchain-google-genai package.

    Args:
        model: Gemini model id (check Google's docs for current names).
        **kw: Passed through to ChatGoogleGenerativeAI. Common options:
            temperature, max_output_tokens.

    YAML example:
        llm:
          type: google_genai
          params:
            model: <gemini model id>
            temperature: 0.1
            max_output_tokens: 800
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, **kw)


@LLMS.register("huggingface_endpoints")
def huggingface_endpoints_llm(repo_id: str, **kw):
    """Create a chat model backed by the HuggingFace Inference API.

    The token is read from the HUGGINGFACEHUB_API_TOKEN environment variable.

    Args:
        repo_id: HuggingFace model id, e.g. "mistralai/Mistral-7B-Instruct-v0.3".
        **kw: Passed through to HuggingFaceEndpoint. Common options:
            temperature, max_new_tokens, streaming.

    YAML example:
        llm:
          type: huggingface_endpoints
          params:
            repo_id: mistralai/Mistral-7B-Instruct-v0.3
            temperature: 0.1
            max_new_tokens: 800
            streaming: true
    """
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    endpoint = HuggingFaceEndpoint(repo_id=repo_id, **kw)
    return ChatHuggingFace(llm=endpoint)


@LLMS.register("huggingface_local")
def huggingface_local_llm(
    model_id: str,
    task: str = "text-generation",
    pipeline_kwargs: dict | None = None,
    **kw,
):
    """Run a HuggingFace model locally with transformers.

    HuggingFacePipeline alone is a text-completion LLM, not a chat model, so it
    is wrapped in ChatHuggingFace to satisfy the registry's BaseChatModel check.

    Args:
        model_id: HuggingFace model id or local path.
        task: Pipeline task. Use "text-generation" for chat models.
        pipeline_kwargs: Generation settings, e.g. {"max_new_tokens": 800,
            "temperature": 0.1, "do_sample": True}.
        **kw: Passed through to HuggingFacePipeline.from_model_id. Common options:
            model_kwargs (passed to from_pretrained, e.g. cache_dir),
            device_map (e.g. "auto").

    YAML example:
        llm:
          type: huggingface_local
          params:
            model_id: mistralai/Mistral-7B-Instruct-v0.3
            task: text-generation
            device_map: auto
            model_kwargs: { cache_dir: ./cache }
            pipeline_kwargs: { max_new_tokens: 800, do_sample: false }
    """
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    pipe = HuggingFacePipeline.from_model_id(
        model_id=model_id, task=task, pipeline_kwargs=pipeline_kwargs or {}, **kw
    )
    return ChatHuggingFace(llm=pipe)


@LLMS.register("ollama")
def ollama_llm(model: str = "llama3.1:8b", **kw):
    """Create a local Ollama chat model (requires a running Ollama server).
    Ollama is a local LLM server that can run models like LLaMA 3. 
    The model name should be in the format "model_name:version".
    
    Args:
        model: Ollama model name (e.g. "llama3.1:8b").
        **kw: Passed through to ChatOllama. Common options:
            temperature, num_predict, top_k, top_p, repeat_penalty.

    YAML example:
        llm:
          type: ollama
          params:
            model: llama3.1:8b
            temperature: 0.1
            num_predict: 800
            top_k: 40
            top_p: 0.9
            repeat_penalty: 1.1   
    """
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, **kw)


@LLMS.register("vllm")
def vllm_llm(model: str, base_url: str = "http://localhost:8000/v1",
             api_key: str = "EMPTY", **kw):
    """Chat model served by vLLM through its OpenAI-compatible API (Stage 7).
    vLLM is a high-performance inference engine for LLMs. It can serve models like LLaMA 3 and others.
    The model name should be in the format "model_name:version".
    Args:
        model: vLLM model name (e.g. "llama3.1:8b").
        base_url: The base URL of the vLLM server's OpenAI-compatible API.
        api_key: The API key for authenticating with the vLLM server.
        **kw: Passed through to ChatOpenAI. Common options:
            temperature, max_tokens, max_retries, timeout.
    
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, **kw)

@LLMS.register("nvidia")
def nvidia_llm(model: str, **kw):
    """Create a chat model served by NVIDIA's AI Foundation Model API.
    
    The API key is read from the NVIDIA_API_KEY environment variable.
    
    Args:
        model: NVIDIA model name (e.g. "meta/llama3-70b-instruct").
        **kw: Passed through to ChatNVIDIA. Common options:
            temperature, max_tokens, top_k, top_p, repetition_penalty.
            
    YAML example:
        llm:
          type: nvidia
          params:
            model: meta/llama3-70b-instruct
            temperature: 0.1
            max_tokens: 800
            top_k: 40
            top_p: 0.9
            repetition_penalty: 1.1
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    return ChatNVIDIA(model=model, **kw)