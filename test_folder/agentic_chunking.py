from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain import hub

llm = OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434", api_key="ollama")

example_text = """
Paragraph 1 - Agricultural Technology:
In recent years, Indian agriculture has seen a rapid integration of technology into traditional farming practices. From the use of drones for monitoring crop health to AI-powered soil analysis tools, farmers are beginning to leverage modern solutions to improve yields. Digital platforms now provide real-time weather updates, crop advisories, and access to financial services, reducing dependency on middlemen and increasing efficiency in supply chains.

Paragraph 2 - Climate Challenges:
Despite technological advances, farmers across the country face severe challenges due to climate change. Irregular monsoons, frequent droughts, and unexpected floods have made traditional farming calendars unreliable. Many regions are experiencing declining groundwater levels and soil fertility, forcing farmers to either adapt with climate-resilient crops or abandon farming altogether. These unpredictable patterns threaten food security and rural livelihoods.

Paragraph 3 - Government Schemes and Policy:
To address these issues, the Indian government has introduced several schemes aimed at empowering the agricultural sector. Initiatives like PM-KISAN offer direct cash transfers to farmers, while the e-NAM platform facilitates digital trade across markets. Additionally, subsidies for fertilizers, seeds, and irrigation equipment are helping marginal farmers adopt better practices. However, policy implementation still faces hurdles such as lack of awareness, bureaucratic delays, and infrastructural gaps in remote areas.
"""

prompt = hub.pull("wfh/proposal-indexing")

chain = prompt | llm

response = chain.invoke({"input": example_text})
print(response)

import re

raw_chunks = re.split(r"(?:Chunk\s*\d+:)", response.content if hasattr(response, "content") else response)
chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

# Step 7: Assign IDs and print results
chunk_data = [{"id": f"chunk-{i+1}", "text": chunk} for i, chunk in enumerate(chunks)]

# Output results
print(f"\n✅ Total chunks created: {len(chunk_data)}\n")
for chunk in chunk_data:
    print(f"🧩 ID: {chunk['id']}\n{chunk['text']}\n{'-'*60}")