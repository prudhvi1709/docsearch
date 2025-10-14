/**
 * Cloudflare Worker for inserting vectors into Vectorize
 * This worker receives embedding data and inserts it into the Vectorize index
 */

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        },
      });
    }

    try {
      const url = new URL(request.url);
      
      if (request.method === 'POST' && url.pathname === '/embed') {
        return await handleEmbedding(request, env);
      }
      
      if (request.method === 'POST' && url.pathname === '/search') {
        return await handleSearch(request, env);
      }
      
      if (request.method === 'GET' && url.pathname === '/health') {
        return new Response(JSON.stringify({ status: 'ok', timestamp: new Date().toISOString() }), {
          headers: { 'Content-Type': 'application/json' },
        });
      }

      return new Response('Not Found', { status: 404 });
    } catch (error) {
      console.error('Worker error:', error);
      return new Response(JSON.stringify({ 
        error: 'Internal Server Error', 
        message: error.message 
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  },
};

async function handleEmbedding(request, env) {
  try {
    const data = await request.json();
    
    // Validate input
    if (!data.vectors || !Array.isArray(data.vectors)) {
      return new Response(JSON.stringify({ 
        error: 'Invalid input', 
        message: 'Expected vectors array' 
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Validate each vector
    for (const vector of data.vectors) {
      if (!vector.id || !vector.values || !Array.isArray(vector.values)) {
        return new Response(JSON.stringify({ 
          error: 'Invalid vector format', 
          message: 'Each vector must have id and values array' 
        }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    console.log(`Inserting ${data.vectors.length} vectors into Vectorize`);

    // Insert vectors into Vectorize
    const results = await env.VECTORIZE_INDEX.insert(data.vectors);
    
    console.log('Insert results:', results);

    return new Response(JSON.stringify({ 
      success: true, 
      inserted: results.count || data.vectors.length,
      message: `Successfully inserted ${data.vectors.length} vectors`
    }), {
      status: 200,
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });

  } catch (error) {
    console.error('Embedding error:', error);
    return new Response(JSON.stringify({ 
      error: 'Embedding failed', 
      message: error.message 
    }), {
      status: 500,
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }
}

async function handleSearch(request, env) {
  try {
    const data = await request.json();
    
    // Validate input
    if (!data.q) {
      return new Response(JSON.stringify({ 
        error: 'Invalid input', 
        message: 'Query parameter "q" is required' 
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const query = data.q;
    const limit = data.ndocs || 15;  // Default to 15 as requested

    console.log(`Searching for: "${query}" (limit: ${limit})`);

    // Generate embedding for the search query using OpenAI
    const queryEmbedding = await generateEmbedding(query, env);
    
    // Query Vectorize index
    const results = await env.VECTORIZE_INDEX.query(queryEmbedding, { 
      topK: limit,
      returnValues: false,
      returnMetadata: true
    });

    console.log(`Found ${results.matches?.length || 0} matches`);

    // Format results for the client
    const documents = results.matches?.map(match => ({
      id: match.id,
      score: match.score,
      metadata: match.metadata || {}
    })) || [];

    // Generate summary based on retrieved documents
    let summary = null;
    if (documents.length > 0) {
      summary = await generateSummary(query, documents, env);
    }

    return new Response(JSON.stringify({ 
      query: query,
      documents: documents,
      totalResults: documents.length,
      summary: summary
    }), {
      status: 200,
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });

  } catch (error) {
    console.error('Search error:', error);
    return new Response(JSON.stringify({ 
      error: 'Search failed', 
      message: error.message 
    }), {
      status: 500,
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }
}

async function generateEmbedding(text, env) {
  // Get OpenAI API key from environment variable
  const openaiApiKey = env.OPENAI_API_KEY;
  
  console.log('Environment keys:', Object.keys(env));
  console.log('OpenAI key exists:', !!openaiApiKey);
  
  if (!openaiApiKey) {
    throw new Error('OpenAI API key not configured in worker environment');
  }

  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openaiApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: text,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.data[0].embedding;
}

async function generateSummary(query, documents, env) {
  // Get OpenAI API key from environment variable
  const openaiApiKey = env.OPENAI_API_KEY;
  
  
  if (!openaiApiKey) {
    throw new Error('OpenAI API key not configured in worker environment');
  }

  // Prepare context from retrieved documents
  const context = documents.map((doc, index) => {
    const preview = doc.metadata.content_preview || '';
    const filename = doc.metadata.filename || `Document ${index + 1}`;
    return `**${filename}** (Score: ${doc.score.toFixed(3)}):\n${preview}`;
  }).join('\n\n');

  // Create prompt for summary generation
  const prompt = `Based on the following documents, please provide a comprehensive answer to the user's question.

User Question: "${query}"

Retrieved Documents:
${context}

Please provide a detailed answer based on the information from these documents. If the documents don't contain enough information to fully answer the question, mention what information is available and what might be missing. Cite the relevant document names when referencing specific information.

Answer:`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openaiApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
       model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful assistant that answers questions based on provided document excerpts. Always cite the document names when referencing specific information.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    console.error('OpenAI summary generation error:', error);
    return {
      text: 'Sorry, I was unable to generate a summary at this time.',
      error: `OpenAI API error: ${response.status}`
    };
  }

  const data = await response.json();
  return {
    text: data.choices[0].message.content.trim(),
       model: 'gpt-4o-mini',
    tokens_used: data.usage?.total_tokens || 0
  };
}