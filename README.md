# MaAS: Multi-agent Architecture Search via Agentic Supernet

## 📰 News

🚩 Updates (2025-2-06) Initial upload to arXiv [PDF](https://arxiv.org/abs/2502.04180).


## 🤔 What is Agentic Supernet?

We *for the first time* shift the paradigm of automated multi-agent system design from seeking a (possibly non-existent) single optimal system to optimizing a probabilistic, continuous distribution of agentic architectures, termed the **agentic supernet**. 

![MaAS](assets/MaAS.png)

## 👋🏻 Method Overview

Building on this concept, we propose **MaAS**, which dynamically samples multi-agent systems that deliver satisfactory performance and token efficiency for user queries across different domains and varying levels of difficulty. Concretely, MaAS takes diverse and varying difficulty queries as input and leverages a controller to sample a subnetwork from the agentic supernet for each query, corresponding to a customized multi-agent system. After the sampled system executes the query, MaAS receives environment feedback and jointly optimizes the supernet’s parameterized distribution and agentic operators.

![framework](assets/framework.png)

## 🏃‍♂️‍➡️ Quick Start

### 📊 Datasets

Please download the  `GSM8K`,  `HumanEval`, `MATH`datasets and place it in the `maas\ext\maas\data` folder. The file structure should be organized as follows:

```
data
└── gsm8k_train.jsonl
└── gsm8k_test.jsonl
└── ......
```

### 🔑 Add API keys

You can configure `~/.metagpt/config2.yaml` according to the example.yaml. Or you can configure `~/config/config2.yaml`.

```python
llm:
  api_type: "openai" 
  model: "gpt-4o-mini" 
  base_url: ""
  api_key: ""
```

### 🐹 Run the code

The code below verifies the experimental results of the `HumanEval` dataset.

```bash
python -m examples.maas.optimize --dataset HumanEval --round 1 --sample 4 --exec_model_name "gpt-4o-mini"
python -m examples.maas.optimize --dataset HumanEval --round 1 --sample 4 --exec_model_name "gpt-4o-mini" --is_test True
```

## 📚 Citation

If you find this repo useful, please consider citing our paper as follows:

```bibtex
@article{zhang2025agentic-supernet,
  title={Multi-agent Architecture Search via Agentic Supernet},
  author={Zhang, Guibin and Niu, Luyang and Fang, Junfeng and Wang, Kun and Bai, Lei and Wang, Xiang},
  journal={arXiv preprint arXiv:2502.04180},
  year={2025}
}
```

## 🙏 Acknowledgement

Special thanks to the following repositories for their invaluable code and prompt.

Our prompt is partially adapted from [ADAS](https://github.com/ShengranHu/ADAS), [AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare/tree/main), and [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow). Our code and operators are partially adapted from [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow).

## 🚀 MaaS API and Dashboard

MaaS now exposes its multi-agent orchestration capabilities through a RESTful API and user-friendly dashboard, allowing you to leverage dynamic multi-agent workflows for your applications.

### 🔌 API Endpoints

The MaaS API provides the following endpoints:

```
POST /api/v1/query
GET /api/v1/workflows
POST /api/v1/workflows
GET /api/v1/workflows/{workflow_id}
DELETE /api/v1/workflows/{workflow_id}
GET /api/v1/benchmarks
POST /api/v1/benchmarks
```

#### Query Endpoint

```
POST /api/v1/query
```

Submit a query to be processed by the multi-agent system:

```json
{
  "query": "Analyze the performance of Tesla stock over the past quarter and suggest investment strategies",
  "workflow_id": "default",  // Optional: Use a specific workflow configuration
  "parameters": {            // Optional: Override default parameters
    "max_tokens": 2000,
    "temperature": 0.7
  }
}
```

For complete API documentation, see the [API Reference](docs/api.md).

### 🖥️ Dashboard Setup

The MaaS Dashboard provides a visual interface for configuring and monitoring agent pipelines.

#### Installation

```bash
# Install dashboard dependencies
cd dashboard
npm install

# Start the dashboard
npm start
```

The dashboard will be available at `http://localhost:3000`.

### ⚙️ LLM Provider Configuration

MaaS supports multiple LLM providers. Configure your preferred provider in `~/.metagpt/config2.yaml`:

```yaml
llm:
  # Available options: "openai", "anthropic", "azure", "gemini", "ollama"
  api_type: "openai" 
  model: "gpt-4o-mini" 
  base_url: ""
  api_key: ""

# Provider-specific configurations
providers:
  azure:
    deployment_id: "your-deployment-id"
    api_version: "2023-05-15"
  anthropic:
    model: "claude-3-opus-20240229"
  gemini:
    model: "gemini-pro"
```

## 🔄 Multi-Agent Orchestration

MaaS orchestrates multiple agents to solve complex problems through dynamic workflows.

### Workflow Configuration

Create custom workflows through the dashboard or API:

1. Define agent roles and responsibilities
2. Configure communication patterns between agents
3. Set decision-making criteria for workflow branching
4. Establish success/failure conditions

Example workflow configuration:

```json
{
  "name": "research_workflow",
  "description": "Research and analysis workflow",
  "agents": [
    {
      "id": "researcher",
      "role": "Information Gatherer",
      "model": "gpt-4o-mini",
      "instructions": "Gather relevant information about the topic"
    },
    {
      "id": "analyst",
      "role": "Data Analyst",
      "model": "gpt-4o",
      "instructions": "Analyze information and identify patterns"
    },
    {
      "id": "writer",
      "role": "Content Creator",
      "model": "claude-3-opus-20240229",
      "instructions": "Create a comprehensive report"
    }
  ],
  "connections": [
    {"from": "researcher", "to": "analyst"},
    {"from": "analyst", "to": "writer"}
  ]
}
```

## 🛠️ Custom Operators

MaaS allows you to create and use custom operators to extend the capabilities of your multi-agent systems.

### Available Operators

- **SearchOperator**: Web search capabilities
- **CodeExecutionOperator**: Execute and validate code
- **DataAnalysisOperator**: Analyze structured data
- **MemoryOperator**: Store and retrieve information
- **ReasoningOperator**: Step-by-step reasoning

### Creating Custom Operators

Extend the `BaseOperator` class to create custom operators:

```python
from maas.operators import BaseOperator

class MyCustomOperator(BaseOperator):
    def __init__(self, name="my_custom_operator"):
        super().__init__(name=name)
        
    async def run(self, context, **kwargs):
        # Implement your operator logic here
        return {"result": "Custom operation completed"}
```

Register your custom operator:

```python
from maas.registry import register_operator

register_operator("my_custom_operator", MyCustomOperator)
```

## 📊 Benchmarking and Optimization

MaaS includes tools for benchmarking and optimizing your multi-agent systems.

### Running Benchmarks

```bash
python -m maas.benchmark --workflow research_workflow --dataset custom_dataset.jsonl --metrics accuracy,latency,cost
```

### Optimization Features

- **Performance Analysis**: Identify bottlenecks in your agent workflows
- **Cost Optimization**: Reduce token usage while maintaining performance
- **A/B Testing**: Compare different workflow configurations
- **Automatic Tuning**: Optimize hyperparameters for specific use cases

View benchmark results in the dashboard or export them for further analysis:

```bash
python -m maas.benchmark --export results.csv
```

## 🔗 Integration Examples

### Python Client

```python
from maas.client import MaaSClient

client = MaaSClient(api_key="your-api-key")

# Submit a query
response = client.query(
    "Analyze the impact of recent Fed policy on tech stocks",
    workflow_id="financial_analysis"
)

print(response.result)
```

### REST API

```bash
curl -X POST https://api.maas.example.com/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query": "Analyze the impact of recent Fed policy on tech stocks",
    "workflow_id": "financial_analysis"
  }'
```
