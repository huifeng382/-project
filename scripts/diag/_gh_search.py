import urllib.request, urllib.parse, json, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get(url):
    r = urllib.request.urlopen(url, timeout=20)
    return json.load(r)

for q in ['TransiLog-share', 'NetlistOpt', 'TransiLog']:
    try:
        d = get('https://api.github.com/search/repositories?q=' + urllib.parse.quote(q) + '&per_page=10')
        print(f'--- 搜索 {q}: {d.get("total_count", 0)} 个 ---')
        for it in d.get('items', []):
            print('  ', it['full_name'], '| 私有:', it['private'], '| 更新:', (it.get('updated_at') or '')[:10],
                  '|', (it.get('description') or '')[:40])
    except Exception as e:
        print(f'搜索 {q} 失败:', e)
