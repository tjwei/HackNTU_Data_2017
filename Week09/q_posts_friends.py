from collections import Counter
from pprint import pprint
count = Counter()
posts = graph.get_object('me', fields=['posts.limit(100)'])['posts']['data']
for i, p in enumerate(posts):
    likes = get_all_data(graph, p['id']+"/likes")
    print(i, p['id'], len(likes))
    for  x in likes:
        name = x['name']
        count[name] += 1
pprint(count.most_common(15))