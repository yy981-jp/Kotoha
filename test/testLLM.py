from llama_cpp import Llama

llm = Llama(
    model_path="Tanuki-8B-dpo-v1.0-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=12,   # CPUコア数に合わせる
    n_gpu_layers=0 # ←これ超重要（完全CPU）
)

def convert(text):
    prompt = f"""次のひらがなの文を、自然な漢字交じりの文に変換してください。
説明は一切書かず、変換結果だけを出力してください。

入力: {text}
出力:"""

    output = llm(
        prompt,
        max_tokens=50,
        temperature=0.0,   # ←変換タスクはこれ重要（ブレ防止）
        stop=["入力:"]
    )

    return output["choices"][0]["text"].strip()


print(convert("きょうはいいてんきですね"))
# print(convert(input("入力: ")))