use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

// ── Core Rust types ──────────────────────────────────────────────────────────

/// Result type alias for the library.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// A chat message with a role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request body sent to the chat completions API.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// A single choice in a non-streaming chat completion response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(rename = "finish_reason")]
    pub finish_reason: Option<String>,
}

/// Usage information from the API response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Non-streaming chat completion response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChatUsage>,
}

/// A single delta in a streaming SSE chunk.
#[derive(Debug, Serialize, Deserialize)]
pub struct StreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

/// A single choice in a streaming SSE chunk.
#[derive(Debug, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    #[serde(rename = "finish_reason")]
    pub finish_reason: Option<String>,
}

/// A single SSE chunk from the streaming chat completions API.
#[derive(Debug, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

/// A stream of SSE chunks. Each item is a parsed `StreamChunk` or an error.
pub type ChatStream = Pin<Box<dyn futures_core::Stream<Item = Result<StreamChunk>> + Send>>;

/// The main client for interacting with an OpenAI-compatible chat API.
#[derive(Debug, Clone)]
pub struct ChatClient {
    base_url: String,
    api_key: String,
    model_name: String,
    http_client: Client,
}

impl ChatClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model_name: model_name.into(),
            http_client: Client::new(),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("BASE_URL").map_err(|_| "BASE_URL environment variable not set")?;
        let api_key = std::env::var("API_KEY").map_err(|_| "API_KEY environment variable not set")?;
        let model_name = std::env::var("MODEL_NAME").map_err(|_| "MODEL_NAME environment variable not set")?;
        Ok(Self::new(base_url, api_key, model_name))
    }

    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatCompletionResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatCompletionRequest { model: &self.model_name, messages, stream: None };

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        if !self.api_key.is_empty() {
            headers.insert(reqwest::header::AUTHORIZATION, format!("Bearer {}", self.api_key).parse()?);
        }

        let response = self.http_client.post(&url).headers(headers).json(&body).send().await?;
        if !response.status().is_success() {
            return Err(format!("API error: {}", response.text().await?).into());
        }
        Ok(response.json().await?)
    }

    pub async fn chat_stream(&self, messages: &[ChatMessage]) -> Result<ChatStream> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatCompletionRequest { model: &self.model_name, messages, stream: Some(true) };

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        if !self.api_key.is_empty() {
            headers.insert(reqwest::header::AUTHORIZATION, format!("Bearer {}", self.api_key).parse()?);
        }

        let response = self.http_client.post(&url).headers(headers).json(&body).send().await?;
        if !response.status().is_success() {
            return Err(format!("API error: {}", response.text().await?).into());
        }

        let byte_stream = response.bytes_stream();
        let stream = async_stream::stream! {
            let mut buffer = String::new();
            for await chunk in byte_stream {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => { yield Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>); continue; }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();
                    if line.is_empty() { continue; }
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" { return; }
                        match serde_json::from_str::<StreamChunk>(data) {
                            Ok(chunk) => yield Ok(chunk),
                            Err(e) => yield Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

// ── Python bindings (only when building with maturin) ────────────────────────

#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyType};
    use std::sync::OnceLock;
    use tokio::runtime::Runtime;

    fn runtime() -> &'static Runtime {
        static RT: OnceLock<Runtime> = OnceLock::new();
        RT.get_or_init(|| Runtime::new().expect("Failed to create tokio runtime"))
    }

    fn value_to_py(py: Python<'_>, v: &serde_json::Value) -> PyObject {
        match v {
            serde_json::Value::Null => py.None(),
            serde_json::Value::Bool(b) => b.to_object(py),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() { i.to_object(py) }
                else if let Some(f) = n.as_f64() { f.to_object(py) }
                else { py.None() }
            }
            serde_json::Value::String(s) => s.to_object(py),
            serde_json::Value::Array(arr) => {
                let list: Vec<PyObject> = arr.iter().map(|x| value_to_py(py, x)).collect();
                list.to_object(py)
            }
            serde_json::Value::Object(obj) => {
                let dict = PyDict::new_bound(py);
                for (k, v) in obj {
                    dict.set_item(k, value_to_py(py, v)).ok();
                }
                dict.into()
            }
        }
    }

    fn to_py_dict<T: Serialize>(py: Python<'_>, val: &T) -> PyResult<PyObject> {
        let json = serde_json::to_value(val)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(value_to_py(py, &json))
    }

    // ── PyChatClient ─────────────────────────────────────────────────────

    #[pyclass(name = "ChatClient")]
    #[derive(Clone)]
    pub struct PyChatClient {
        inner: ChatClient,
    }

    #[pymethods]
    impl PyChatClient {
        #[new]
        #[pyo3(signature = (base_url, api_key, model_name))]
        fn new(base_url: String, api_key: String, model_name: String) -> Self {
            Self { inner: ChatClient::new(base_url, api_key, model_name) }
        }

        #[classmethod]
        fn from_env(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
            let _ = dotenvy::dotenv();
            ChatClient::from_env()
                .map(|c| Self { inner: c })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn model_name(&self) -> &str {
            self.inner.model_name()
        }

        fn chat(&self, py: Python<'_>, messages: Vec<PyObject>) -> PyResult<PyObject> {
            let msgs = py_messages_to_rust(py, &messages)?;
            let resp = runtime().block_on(self.inner.chat(&msgs))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            to_py_dict(py, &resp)
        }

        fn chat_stream(&self, py: Python<'_>, messages: Vec<PyObject>) -> PyResult<PyObject> {
            let msgs = py_messages_to_rust(py, &messages)?;
            let mut stream = runtime().block_on(self.inner.chat_stream(&msgs))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let mut chunks: Vec<PyObject> = Vec::new();
            use futures_util::StreamExt;
            loop {
                match runtime().block_on(stream.as_mut().next()) {
                    Some(Ok(chunk)) => chunks.push(to_py_dict(py, &chunk)?),
                    Some(Err(e)) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
                    None => break,
                }
            }
            Ok(chunks.to_object(py))
        }
    }

    fn py_messages_to_rust(py: Python<'_>, messages: &[PyObject]) -> PyResult<Vec<ChatMessage>> {
        messages.iter().map(|obj| {
            let dict = obj.downcast_bound::<PyDict>(py)
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("Each message must be a dict"))?;
            let role: String = dict.get_item("role")
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'role' key"))?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'role' key"))?
                .extract()?;
            let content: String = dict.get_item("content")
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'content' key"))?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'content' key"))?
                .extract()?;
            Ok(ChatMessage { role, content })
        }).collect()
    }

    // ── Module ───────────────────────────────────────────────────────────

    #[pymodule]
    fn hqagent(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyChatClient>()?;
        Ok(())
    }
}
