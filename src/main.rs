use hqagent::ChatClient;

#[tokio::main]
async fn main() {
    // 加载 .env 文件（不存在时不报错，允许通过系统环境变量设置）
    let _ = dotenvy::dotenv();

    let client = ChatClient::from_env().expect("Missing env vars: BASE_URL, API_KEY, MODEL_NAME.\nCopy .env.example to .env and fill in your values.");

    println!("HQAgent library ready. Model: {}", client.model_name());
}
