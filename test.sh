#!/bin/bash
# Check if the number of requests was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <num_requests>"
    exit 1
fi

# The URL of the endpoint
url="http://0.0.0.0:8000/embeddings/"
url2='http://localhost:8000/rerank/'
# JSON data to send
json_data='{
  "sentences": [
    "This is the first example sentence.",
    "Here goes the second example."
  ]
}'

json_data2='{
  "sentence_pairs": [
    ["Oraci  n 1 del par 1", "Oraci  n 2 del par 1"],
    ["Oraci  n 1 del par 2", "Oraci  n 2 del par 2"],
    ["Oraci  n 1 del par 3", "Oraci  n 2 del par 3"]
  ]
}'

# Number of concurrent requests, read from the first script argument
num_requests=$1
start_time=$(date +%s)
echo "Test mix embeddings and rerank"
for i in $(seq 1 $num_requests); do
  if [ $((RANDOM % 2)) -eq 0 ]; then
    curl -X POST "$url" \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -d "$json_data" &
  else
    curl -X POST "$url2" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d "$json_data2" &
  fi
done
wait
echo "All requests have been sent."
# Captura el tiempo de finalización
end_time=$(date +%s)

# Calcula la diferencia y la imprime
duration=$((end_time - start_time))
echo "Total time taken: $duration seconds"
