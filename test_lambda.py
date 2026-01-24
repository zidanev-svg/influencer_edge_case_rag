import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from lambda_handler import lambda_handler

@pytest.fixture
def mock_openai_response():
    mock_embedding = [0.1] * 1536
    return mock_embedding

@pytest.fixture
def mock_pinecone_response():
    return {
        'matches': [
            {
                'id': 'EDGE-004',
                'score': 0.95,
                'metadata': {
                    'influencer_id': 'EDGE-004',
                    'platform': 'Instagram',
                    'content_type': 'Reel',
                    'content_category': 'Fashion & Beauty',
                    'content_subcategory': 'Fashion Hauls',
                    'creator_country': 'US',
                    'rate_amount': 75000,
                    'currency': 'USD',
                    'followers': 3200000,
                    'engagement_rate_pct': 4.8,
                    'edge_case_notes': 'Top-tier influencer with premium rate',
                    'edge_case_questions': 'Is $75K justified?'
                }
            },
            {
                'id': 'EDGE-005',
                'score': 0.88,
                'metadata': {
                    'influencer_id': 'EDGE-005',
                    'platform': 'Instagram',
                    'content_type': 'Reel',
                    'content_category': 'Fashion & Beauty',
                    'content_subcategory': 'Style Tips',
                    'creator_country': 'US',
                    'rate_amount': 68000,
                    'currency': 'USD',
                    'followers': 2800000,
                    'engagement_rate_pct': 5.1,
                    'edge_case_notes': 'Premium influencer pricing',
                    'edge_case_questions': 'Should we negotiate?'
                }
            }
        ]
    }

@patch('lambda_handler.index')
@patch('lambda_handler.client')
def test_lambda_handler_success(mock_client, mock_index, mock_openai_response, mock_pinecone_response):
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=mock_openai_response)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    mock_index.query.return_value = mock_pinecone_response
    
    event = {
        'body': json.dumps({
            'query': 'Instagram Reel influencer, $80K, Fashion & Beauty, US',
            'top_k': 3
        })
    }
    
    response = lambda_handler(event, None)
    
    assert response['statusCode'] == 200
    body = json.loads(response['body'])
    assert 'matches' in body
    assert len(body['matches']) == 2
    assert body['matches'][0]['influencer_id'] == 'EDGE-004'
    assert body['matches'][0]['similarity_score'] == 0.95

@patch('lambda_handler.index')
@patch('lambda_handler.client')
def test_lambda_handler_missing_query(mock_client, mock_index):
    event = {
        'body': json.dumps({
            'top_k': 3
        })
    }
    
    response = lambda_handler(event, None)
    
    assert response['statusCode'] == 400
    body = json.loads(response['body'])
    assert 'error' in body

@patch('lambda_handler.index')
@patch('lambda_handler.client')
def test_lambda_handler_default_top_k(mock_client, mock_index, mock_openai_response, mock_pinecone_response):
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=mock_openai_response)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    mock_index.query.return_value = mock_pinecone_response
    
    event = {
        'body': json.dumps({
            'query': 'Test query'
        })
    }
    
    response = lambda_handler(event, None)
    
    assert response['statusCode'] == 200
    mock_index.query.assert_called_once()
    call_args = mock_index.query.call_args[1]
    assert call_args['top_k'] == 3

@patch('lambda_handler.index')
@patch('lambda_handler.client')
def test_lambda_handler_error_handling(mock_client, mock_index):
    mock_client.embeddings.create.side_effect = Exception('OpenAI API error')
    
    event = {
        'body': json.dumps({
            'query': 'Test query',
            'top_k': 3
        })
    }
    
    response = lambda_handler(event, None)
    
    assert response['statusCode'] == 500
    body = json.loads(response['body'])
    assert 'error' in body

@patch('lambda_handler.index')
@patch('lambda_handler.client')
def test_lambda_handler_cors_headers(mock_client, mock_index, mock_openai_response, mock_pinecone_response):
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=mock_openai_response)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    mock_index.query.return_value = mock_pinecone_response
    
    event = {
        'body': json.dumps({
            'query': 'Test query',
            'top_k': 3
        })
    }
    
    response = lambda_handler(event, None)
    
    assert 'Access-Control-Allow-Origin' in response['headers']
    assert response['headers']['Access-Control-Allow-Origin'] == '*'

@patch('lambda_handler.index')
@patch('lambda_handler.client')
def test_lambda_handler_custom_top_k(mock_client, mock_index, mock_openai_response, mock_pinecone_response):
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=mock_openai_response)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    mock_index.query.return_value = mock_pinecone_response
    
    event = {
        'body': json.dumps({
            'query': 'Test query',
            'top_k': 5
        })
    }
    
    response = lambda_handler(event, None)
    
    assert response['statusCode'] == 200
    call_args = mock_index.query.call_args[1]
    assert call_args['top_k'] == 5
