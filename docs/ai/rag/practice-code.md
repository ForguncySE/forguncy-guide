# RAG 实战 - 代码模式

本章节将通过代码实战的方式，帮助您快速搭建一个 RAG 系统。教程会使用三种不同的技术栈，分别是：
- Python(3.12 及以上), 使用 `pdm` 作为包管理器。
- Node.js(18.x 及以上), 使用 `npm` 作为包管理器。
- Java(JDK 1.8 及以上), 使用 `maven` 作为包管理器。

可以按需选择其中一种技术栈进行实战。

## 环境准备

### 向量数据库
为保证多技术栈的依赖一致性，我们选择 [`Qdrant`](https://qdrant.tech) 作为向量数据库。

> [!TIP]
> Qdrant 暂不支持内存存储，因此建议学习时可选择 [docker](https://qdrant.tech/documentation/quickstart/#download-and-run) 的方式进行 Qdrant 服务的部署。

Qdrant 服务部署完成后，我们需要使用 Qdrant 客户端与其进行交互。

::: code-group
```sh [python]
pdm add qdrant-client
```
```sh [node]
npm install @qdrant/js-client-rest
```
```xml [java]
<dependency>
  <groupId>io.qdrant</groupId>
  <artifactId>client</artifactId>
  <version>1.15.0</version>
</dependency>
```
:::

### 模型服务
教程会使用向量模型和标准大语言模型。为方便用户体验，我们会使用阿里云百炼平台的模型。

- 向量模型：[`text-embedding-v4`](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.623747bbtjHUF3&tab=model#/model-market/detail/text-embedding-v4)，用于将文本转换为向量。
- 大语言模型：[`qwen-plus`](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.623747bbtjHUF3&tab=model#/model-market/detail/qwen-plus)，用于生成最终回复的内容。

使用模型服务，需要提前配置环境变量 `DASHSCOPE_API_KEY`，并将其设置为您的阿里云百炼平台的 api key。配置方法可参考 [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/configure-api-key-through-environment-variables)。

> [!TIP]
> 您也可以在对应的工程下创建配置文件，从配置文件中读取环境变量。

### 文本准备

可自行准备用于 RAG 的内容文档，用户提问后，RAG 系统会从该文档中进行答案搜索，为方便学习，建议准备文档格式为 `Markdown`。

## RAG 实现

### 分片

将文档文件按段落分割成文本块列表。

该函数读取指定的文档文件，并根据双换行符 `\n\n` 将文档内容分割成多个文本块。

> [!TIP]
> 示例文档的文档类型是 `Markdown`，段落之间会用空行区分，即双换行 `\n\n`。如果您的文档使用其他分隔符，需要根据实际情况修改该函数。

::: code-group

```python [python]
from typing import List

def split_into_chunks(doc_file: str) -> List[str]:
    """
    将文档文件按段落分割成文本块列表
    Args:
        doc_file (str): 要读取的文档文件路径
    Returns:
        List[str]: 包含所有非空文本块的列表，每个元素代表文档中的一个段落或文本块
    """
    with open(doc_file, 'r', encoding='utf-8') as f:
        doc = f.read()
    return [chunk for chunk in doc.split('\n\n')]

```

```js [node]
/**
 * 将文档文件按段落分割成文本块列表
 * @param {string} docFile - 要读取的文档文件路径
 * @returns {Promise<string[]>} 包含所有非空文本块的数组，每个元素代表文档中的一个段落或文本块
 */
async function splitIntoChunks(docFile) {
    const fs = require('fs').promises;
    try {
        const doc = await fs.readFile(docFile, 'utf-8');
        return doc.split('\n\n').filter(chunk => chunk.trim() !== '');
    } catch (error) {
        throw new Error(`Error reading file: ${error.message}`);
    }
}
```

```java [java]
/**
 * 将文档文件按段落分割成文本块列表
 *
 * @param docFile 要读取的文档文件路径
 * @return 包含所有非空文本块的列表，每个元素代表文档中的一个段落或文本块
 * @throws IOException 当文件读取失败时抛出
 */
public static List<String> splitIntoChunksByParagraph(String docFile) throws IOException {
    String doc = new String(Files.readAllBytes(Paths.get(docFile)));
    String[] chunks = doc.split("\n\n");
    List<String> result = new ArrayList<>();

    for (String chunk : chunks) {
        if (!chunk.trim().isEmpty()) {
            result.add(chunk);
        }
    }
    return result;
}

// ======== 分片 ========
List<String> chunks = TextUtil.splitIntoChunksByParagraph("src/main/resources/doc.md");

```
:::

### 索引

1. 使用 embedding 模型将第一步切割好的文本块依次转换为对应的文本向量。

::: code-group

```python [python]
from http import HTTPStatus
import dashscope

# 用于获取文本的向量表示
def get_embedding(text: str, model: str = "text-embedding-v4", dimension: int = 1024) -> list[float]:
    resp = dashscope.TextEmbedding.call(
        model=model,
        input=text,
        dimension=dimension,
        output_type="dense&sparse"
    )

    if resp.status_code == HTTPStatus.OK:
    embeddings_list = resp["output"]["embeddings"]
    dense_embedding = embeddings_list[0]["embedding"]
    return dense_embedding

# 将上一步切割好的文本块转换为向量
embeddings = [get_embedding(chunk) for chunk in chunks]
```
```js [node]
import dotenv from 'dotenv';
import OpenAI from "openai";

// 将.env 文件中的变量加载到 process.env 中
dotenv.config()
// 初始化 openai 客户端
const openai = new OpenAI({
    apiKey: process.env.DASHSCOPE_API_KEY, // 从环境变量读取
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1'
});

async function getEmbeddings(inputContent) {
    try {
        const resp = await openai.embeddings.create({
            model: "text-embedding-v4",
            input: inputContent,
            dimensions: 1024,
            encoding_format: "float"
        });
        if (resp && resp.data && Array.isArray(resp.data)) {
            const embeddings = resp.data.map(item => item.embedding);
            return embeddings;
        }
    } catch (error) {
        console.error('Error:', error);
    }
}
```
```java [java]
/**
* 从配置文件中加载API Key
*/
private void loadApiKey() {
    try (InputStream input = getClass().getClassLoader().getResourceAsStream("application.properties")) {
        Properties prop = new Properties();
        prop.load(input);
        apiKey = prop.getProperty("dashscope.api-key");
    } catch (IOException e) {
        throw new RuntimeException("Failed to load API key from configuration", e);
    }
}

/**
 * 生成文本嵌入向量
 * @param textList 文本列表
 * @return 向量结果
 */
public List<TextEmbeddingResultItem> textEmbedding(List<String> textList) throws ApiException {
    TextEmbeddingParam param = null;
    TextEmbedding textEmbedding = null;
    try {
        param = TextEmbeddingParam
                .builder()
                .apiKey(apiKey)
                .model("text-embedding-v4")
                .texts(textList)
                .parameter("dimension", 1024)
                .outputType(TextEmbeddingParam.OutputType.DENSE_AND_SPARSE)
                .build();
        textEmbedding = new TextEmbedding();
        TextEmbeddingResult result = textEmbedding.call(param);
        return result.getOutput().getEmbeddings();
    } catch (ApiException | NoApiKeyException e) {
        System.out.println("调用失败：" + e.getMessage());
    }
    return List.of();
}

// ======== 文本向量化 ========
DashScopeClient client = new DashScopeClient();
List<List<Float>> denseEmbeddings = client.textEmbedding(chunks).stream()
            .map(TextEmbeddingResultItem::getEmbedding)
            .map(innerList -> innerList.stream().map(Double::floatValue).toList())
            .toList();
```
:::

2. 将生成的向量存入向量数据库 `Qdrant` 中。

::: code-group

```python [python]
from qdrant_client import QdrantClient, models

# 初始化客户端
qdrant_client = QdrantClient(host="localhost", port=6333)
# 自定义的向量集合名称
COLLECTION_NAME = "rag_python"

def ensure_collection_exists() -> None:
    """
    确保集合存在，如果不存在则创建它。
    """
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        # 如果集合不存在，捕获异常并创建它
        print(f"[INFO] 集合 '{COLLECTION_NAME}' 不存在，正在创建...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )
        print(f"[INFO] 集合 '{COLLECTION_NAME}' 创建成功")
        
def save_embeddings_by_qdrant(chunks: List[str], embeddings: List[List[float]]) -> None:
    """
    将文本块和对应的嵌入向量保存到 Qdrant 集合中。
    
    Args:
        chunks: 文本块列表
        embeddings: 对应的嵌入向量列表
    """

    # 构造 points 列表，每个 point 包含 id、vector 和 payload（payload 中可存文档内容）
    points = [
        models.PointStruct(
            id=i, 
            vector=embedding,
            payload={"text": chunk}  # 将文本存入 payload 中方便后续检索
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    # 确保集合存在
    ensure_collection_exists()
    # 向 Qdrant 集合中上传 points
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

save_embeddings_by_qdrant(chunks, embeddings)
```
```js [node]
import {QdrantClient} from '@qdrant/js-client-rest';

const client = new QdrantClient({url: 'http://127.0.0.1:6333'});

const COLLECTION_NAME = 'rag_node';

/**
 * 确保集合存在，如果不存在则创建它。
 */
async function ensureCollectionExists() {
    try {
        await client.getCollection(COLLECTION_NAME);
    } catch (e) {
        console.log(`[INFO] 集合 '${COLLECTION_NAME}' 不存在，正在创建...`);
        await client.createCollection(COLLECTION_NAME, {
            vectors: {
                size: 1024,
                distance: 'Cosine',
            },
        });
        console.log(`[INFO] 集合 '${COLLECTION_NAME}' 创建成功`);
    }
}

/**
 * 将文本块和对应的嵌入向量保存到 Qdrant 集合中。
 *
 * @param {string[]} chunks - 文本块数组
 * @param {number[][]} embeddings - 对应的嵌入向量数组
 */
async function saveEmbeddings(chunks, embeddings) {
    const points = chunks.map((chunk, index) => ({
        id: index, // 使用数字 ID
        vector: embeddings[index],
        payload: {
            text: chunk, // 将文本存入 payload 中方便后续检索
        },
    }));
    await ensureCollectionExists();
    await client.upsert(COLLECTION_NAME, {
        points: points,
    });
    console.log(`[INFO] 成功上传 ${points.length} 个向量到集合 '${COLLECTION_NAME}'`);
}

await saveEmbeddings(chunks, await embeddings)
```
```java [java]
/**
 * 确保集合存在
 */
public void ensureCollectionExists() throws ExecutionException, InterruptedException {
    try {
        qdrantClient.getCollectionInfoAsync(COLLECTION_NAME).get();
    } catch (Exception e) {
        if (e.getCause() instanceof io.grpc.StatusRuntimeException statusException) {
            if (statusException.getStatus().getCode() == io.grpc.Status.Code.NOT_FOUND) {
                qdrantClient.createCollectionAsync(COLLECTION_NAME,
                                Collections.VectorParams.newBuilder()
                                        .setDistance(Collections.Distance.Cosine)
                                        .setSize(1024)
                                        .build())
                        .get();
            }
        }
    }
}

/**
 * 保存向量
 *
 * @param chunks      文本块列表
 * @param embeddings  嵌入向量列表
 */
public void saveEmbeddings(List<String> chunks, List<List<Float>> embeddings) throws ExecutionException, InterruptedException {
    // 构建 Points 列表
    List<Points.PointStruct> points = IntStream.range(0, chunks.size())
            .mapToObj(i -> {
                List<Float> vector = embeddings.get(i);
                Map<String, JsonWithInt.Value> payload = new HashMap<>();
                payload.put("text", ValueFactory.value(chunks.get(i)));

                return Points.PointStruct.newBuilder()
                        .setId(id(i))
                        .setVectors(VectorsFactory.vectors(vector))
                        .putAllPayload(payload)
                        .build();
            })
            .collect(Collectors.toList());
    // 确保集合存在
    ensureCollectionExists();
    // 批量插入 points
    qdrantClient.upsertAsync(
            COLLECTION_NAME,
            points
    ).get();
    System.out.println("[INFO] 成功上传 " + points.size() + " 个向量到集合 '" + COLLECTION_NAME + "'");
}

// ======== 存储向量与文本 ========
QdrantAgent qdrantAgent = new QdrantAgent();
qdrantAgent.saveEmbeddings(chunks, denseEmbeddings);
```
:::

至此，RAG 实现的索引部分就完成了。

### 召回与重排

我们可以将用户的问题转换为向量，然后从向量数据库中进行相关性的查询。

> [!TIP]
> 由于我们使用的是阿里云百炼的专业向量模型，其处理逻辑对于相关性提供了较好的支持。因此，我们可以直接使用向量模型的输出结果进行召回，而无需进行额外的重排。
> 
> 如果您发现召回的文档与用户问题的相关性较低，您可以尝试调整召回的文档数量 `top_k`，或者使用更专业的向量模型增加二次重排的操作。

::: code-group


```python [python]
# 将用户问题进行向量化，并从向量数据库中查询相关文档
def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = get_dense_embedding(query)
    query_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    )
    retrieved_texts = [point.payload.get("text", "") for point in query_result.points]
    return retrieved_texts

query="请替换为用户的真实问题"
retrieved_chunks = retrieve(query, 5)
```
```js [node]
/**
 * 查询向量
 *
 * @param queryEmbedding 查询向量
 * @param topK           返回的向量数量
 * @return 查询结果
 */
async function queryEmbeddings(queryEmbedding, topK = 5) {
    const res = await client.query(COLLECTION_NAME,
        {
            query: queryEmbedding,
            limit: topK,
            with_payload: true,
        })
    return res.points.map(point => point.payload.text);
}

const query = "请替换为用户的真实问题"
const query_embedding = await getEmbeddings(query)
const query_results = await queryEmbeddings(query_embedding[0], 5)
console.log(query_results);
```
```java [java]
/**
 * 查询向量
 *
 * @param queryEmbedding 查询向量
 * @param topK           返回的向量数量
 * @return 查询结果
 */
public List<String> query(List<Float> queryEmbedding, int topK) throws ExecutionException, InterruptedException {
    List<Points.ScoredPoint> queryResult = qdrantClient.searchAsync(
            Points.SearchPoints.newBuilder()
                    .setCollectionName(COLLECTION_NAME)
                    .addAllVector(queryEmbedding)
                    .setLimit(topK)
                    .setWithPayload(Points.WithPayloadSelector.newBuilder().setEnable(true).build())
                    .build()
    ).get();
    return queryResult.stream().map(point -> point.getPayloadMap().get("text").getStringValue()).toList();
}

// ======== 召回 ========
String query = "请替换为用户的真实问题";
List<List<Float>> queryEmbeddings = client.textEmbedding(List.of(query)).stream()
        .map(TextEmbeddingResultItem::getEmbedding)
        .map(innerList -> innerList.stream().map(Double::floatValue).toList())
        .toList();
List<String> relatedChunks = qdrantAgent.query(queryEmbeddings.getFirst(),5);
```
:::

### 生成

生成阶段需要一个标准的文本生成大模型，将检索出的文档内容进行整理，并输出最终答案。这里选择的是阿里百炼平台提供的文本生成模型 [`qwen-plus`](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.623747bbtjHUF3&tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712576.html)。

::: code-group

```python [python]
import os
import dashscope

def generate(query: str, chunks: List[str]) -> str:
    prompt = f""" 你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。
    
    用户问题：{query}

    相关片段：
    {"\n\n".join(chunks)}
    
    请基于上述内容作答，不要编造信息。如果相关片段中没有相关信息，回答“没有相关信息”。"""

    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': query}
    ]

    response = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-plus",
        messages=messages,
        result_format='message'
    )
    return response['output']['choices'][0]['message']['content']

response = generate(query, retrieved_chunks)
```
```js [node]
/**
 * 调用文本生成模型
 *
 * @param query  用户问题
 * @param prompt 系统提示
 * @return 生成结果
 */
async function callWithMessage(query, prompt) {
    const completion = await openai.chat.completions.create({
        model: "qwen-plus",
        messages: [
            { role: "system", content: prompt},
            { role: "user", content: query }
        ],
    });
    return completion.choices[0].message.content;
}

// 将上一步得到的数组转换为字符串
const related_chunks = query_results.join("\n\n")
const prompt =
    `你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。\n\n` +
    `用户问题：${query}\n\n` +
    `相关片段：\n${related_chunks}\n\n` +
    `请基于上述内容作答，不要编造信息。如果相关片段中没有相关信息，回答“没有相关信息”。`;

const answer = await callWithMessage(query, prompt)
console.log(answer)
```
```java [java]
/**
 * 调用文本生成模型
 *
 * @param query  用户问题
 * @param prompt 系统提示
 * @return 生成结果
 */
public GenerationResult callWithMessage(String query, String prompt) throws ApiException, NoApiKeyException, InputRequiredException {
    Generation gen = new Generation();
    Message systemMsg = Message.builder()
            .role(Role.SYSTEM.getValue())
            .content(prompt)
            .build();
    Message userMsg = Message.builder()
            .role(Role.USER.getValue())
            .content(query)
            .build();
    GenerationParam param = GenerationParam.builder()
            .apiKey(apiKey)
            .model("qwen-plus")
            .messages(Arrays.asList(systemMsg, userMsg))
            .resultFormat(GenerationParam.ResultFormat.MESSAGE)
            .build();
    return gen.call(param);
}

// ======== 生成 ========
String prompt = String.format(
    """
            你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。
            
            用户问题：%s
            
            相关片段：
            %s
            
            请基于上述内容作答，不要编造信息。如果相关片段中没有相关信息，回答“没有相关信息”。""",
    query,
    relatedChunks
);
GenerationResult result = client.callWithMessage(query, prompt);
System.out.println(result.getOutput().getChoices().getFirst().getMessage().getContent());
```

## 总结
至此，我们使用代码开发的方式，完成了一个基于 RAG 的问答系统的完整流程，包括文档索引、问题召回和答案生成。在实际应用中，你可能需要根据具体的业务场景和数据特征进行优化，例如调整召回策略、优化生成模型的提示词等。

接下来，我们可以了解低代码方式的 RAG 系统构建。