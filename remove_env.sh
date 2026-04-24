for env in llara llaradino_clone llaravima qwen25vl robocasa test1 videoagent videollava vidtwo vllm vllm85 vllm90 vllm_env wj; do
    conda env remove -n "$env" -y
done