from collections import  Counter
friends = graph.get_object('me', fields=['friends.limit(1000)'])['friends']['data']
stat = Counter()
for f in friends:
    key = "favorite_athletes"
    obj = graph.get_object(f['id'], fields=['id','name', key])
    if key in obj:
        print(f['id'], f['name'])
        for item in obj[key]:
            print("["+item['name']+"]", end=" ") 
            stat[item['name']] += 1
        print()
pprint(stat.most_common(15))