import EmbedQ

glove = EmbedQ.load_glove()

query = "I'm so excited for the trip next week!"

query_embedded = EmbedQ.embed_text(query, glove)

