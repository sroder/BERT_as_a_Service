from bert_serving.client import BertClient
bc = BertClient()
sent1 = bc.encode(['Java, C, C#'])
sent2 = bc.encode(['Programierungsprache ist sehr wichtig in IT Umwelt.'])

print(len(sent1),'#length of first sentence')


from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(sent1,sent2)[0])
