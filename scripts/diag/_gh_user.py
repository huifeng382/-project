import urllib.request, json, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get(url):
    try:
        r = urllib.request.urlopen(url, timeout=20)
        return json.load(r)
    except Exception as e:
        return {'_err': str(e)}

# 用户 Boucii 是否存在
u = get('https://api.github.com/users/Boucii')
if '_err' in u:
    print('用户 Boucii:', u['_err'])
else:
    print('用户 Boucii 存在:', u.get('login'), '| 公开仓库数:', u.get('public_repos'))
    # 该用户的公开仓库
    repos = get('https://api.github.com/users/Boucii/repos?per_page=100')
    if isinstance(repos, list):
        print('Boucii 的公开仓库:')
        for r in repos:
            print('  ', r['name'], '|', (r.get('description') or '')[:40])
    else:
        print('  repos 查询:', repos.get('_err'))

# 直接再看 TransiLog-share（带大小写变体）
for path in ['Boucii/TransiLog-share', 'Boucii/TransiLogShare', 'boucii/TransiLog-share']:
    d = get(f'https://api.github.com/repos/{path}')
    if '_err' in d:
        print(f'{path}: {d["_err"]}')
    else:
        print(f'{path}: OK, 默认分支={d.get("default_branch")}, 更新={d.get("updated_at","")[:10]}, 私有={d.get("private")}')
