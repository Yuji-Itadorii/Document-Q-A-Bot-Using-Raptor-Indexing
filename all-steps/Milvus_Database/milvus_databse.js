const { MilvusClient } = require("@zilliz/milvus2-sdk-node");

const milvusClient = new MilvusClient("localhost:19530");

async function main() {
  try {
    // Get a list of all collections
    const collections = await milvusClient.listCollections();
    console.log("Collections:", collections);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
