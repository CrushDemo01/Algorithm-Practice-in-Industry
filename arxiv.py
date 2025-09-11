
'''
credit to original author: Glenn (chenluda01@outlook.com)
Author: Doragd
'''

import os
import requests
import time
import json
import datetime
from tqdm import tqdm
from translate import translate, init_model_client

SERVERCHAN_API_KEY = os.environ.get("SERVERCHAN_API_KEY", None)
QUERY = os.environ.get('QUERY', 'cs.IR')
LIMITS = int(os.environ.get('LIMITS', 3)) + 10
FEISHU_URL = os.environ.get("FEISHU_URL", None)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "DeepSeek")
PROMPT = os.environ.get("PROMPT", '无')
MIN_SCORE = os.environ.get('MIN_SCORE', 6)
try:
    MIN_SCORE = float(MIN_SCORE) if MIN_SCORE is not None else None
except Exception:
    MIN_SCORE = None

def get_yesterday():
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')


def search_arxiv_papers(search_term, max_results=10):
    papers = []

    url = f'http://export.arxiv.org/api/query?' + \
          f'search_query=all:{search_term}' +  \
          f'&start=0&&max_results={max_results}' + \
          f'&sortBy=submittedDate&sortOrder=descending'

    response = requests.get(url)

    if response.status_code != 200:
        return []

    feed = response.text
    entries = feed.split('<entry>')[1:]

    if not entries:
        return []

    print('[+] 开始处理每日最新论文....')

    for entry in entries:

        title = entry.split('<title>')[1].split('</title>')[0].strip()
        summary = entry.split('<summary>')[1].split('</summary>')[0].strip().replace('\n', ' ').replace('\r', '')
        url = entry.split('<id>')[1].split('</id>')[0].strip()
        pub_date = entry.split('<published>')[1].split('</published>')[0]
        pub_date = datetime.datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

        papers.append({
            'title': title,
            'url': url,
            'pub_date': pub_date,
            'summary': summary,
            'translated': '',
        })
    
    print('[+] 开始翻译每日最新论文并缓存....')

    papers = save_and_translate(papers)
    
    return papers


def send_wechat_message(title, content, SERVERCHAN_API_KEY):
    url = f'https://sctapi.ftqq.com/{SERVERCHAN_API_KEY}.send'
    params = {
        'title': title,
        'desp': content,
    }
    requests.post(url, params=params)

def send_feishu_message(title, content, url=FEISHU_URL):
    if not url:
        raise Exception("未设置FEISHU_URL环境变量或参数")

    # 交互卡片（无图片，避免 img_key 失败），失败则回退纯文本
    card_data = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green",
            "title": {"tag": "plain_text", "content": title[:250]}
        },
        "elements": [
            {"tag": "markdown", "content": content[:25000]}
        ]
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url=url, data=json.dumps({
            "msg_type": "interactive",
            "card": card_data
        }), headers=headers)
        ok = (resp.status_code == 200)
        try:
            ret = resp.json()
            ok = ok and (ret.get("code") == 0)
        except Exception:
            ok = False
        if ok:
            return
    except Exception:
        pass

    # 回退：纯文本
    fallback = {
        "msg_type": "text",
        "content": {"text": f"{title}\n{content}"[:25000]}
    }
    requests.post(url=url, data=json.dumps(fallback), headers=headers)


def save_and_translate(papers, filename='arxiv.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)

    cached_title2idx = {result['title'].lower():i for i, result in enumerate(results)}
    
    untranslated_papers = []
    translated_papers = []
    for paper in papers:
        title = paper['title'].lower()
        if title in cached_title2idx.keys():
            translated_papers.append(
                results[cached_title2idx[title]]
            )
        else:
            untranslated_papers.append(paper)
    
    source = []
    for paper in untranslated_papers:
        source.append(paper['summary'])
    target = translate(source)
    if len(target) == len(untranslated_papers):
        for i in range(len(untranslated_papers)):
            untranslated_papers[i]['translated'] = target[i]
    
    results.extend(untranslated_papers)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f'[+] 总检索条数: {len(papers)} | 命中缓存: {len(translated_papers)} | 实际返回: {len(untranslated_papers)}....')

    return untranslated_papers # 只需要发送缓存中没有的

def rank_papers_with_deepseek(papers, top_k=None, min_score: float = None, temperature=0.4):
    """
    输入: papers = [{"title","summary","url","pub_date","translated",...}, ...]
    输出: 仅保留 top_k 的子集，并附加 score/decision/reason 字段
    """
    import json as _json
    import re as _re

    if not papers:
        return []

    client = init_model_client()
    system_prompt = {
        "role": "system",
        "content": (
            "你是一位资深学术评审，擅长在 人工智能 方向审稿高质量论文。\n"
            "综合考虑：创新性、方法有效性、实证力度、潜在影响力、与工业可落地性。\n"
            "请给每篇论文 0-10 的评分（允许小数），并挑选值得关注的论文。"
            f"请严格限制要求<要求>以下主题的论文，不相关的请score置为0。请严格限制要求<要求>以下主题的论文，不相关的请score置为0。请严格限制要求<要求>以下主题的论文，不相关的请score置为0。<要求>：{PROMPT}。<\要求>\n"
            "仅输出 JSON 数组，不要多余文本。每个元素包含：\n"
            '{"index": 整数, "score": 浮点数, "decision": "keep|drop", "reason": "简短中文理由"}'
        )
    }

    items = []
    # 控制 token 成本：限制标题/摘要长度，并优先提供中文译文辅助判断
    def _truncate(text: str, max_len: int) -> str:
        return text[:max_len] + ("…" if len(text) > max_len else "")

    for i, p in enumerate(papers):
        title = _truncate(p.get("title", "").strip(), 200)
        zh = p.get("translated", "").strip()
        summary = p.get("summary", "").strip()
        if zh:
            body = _truncate(zh, 1200)
        else:
            body = _truncate(summary, 1200)
        items.append(f"{i}. {title}\n{body}")

    user_prompt = (
        f"下面是待评审论文，共 {len(papers)} 篇。请选筛选：\n" +
        "\n".join(items) +
        "\n请仅返回 JSON 数组（UTF-8，无额外注释/代码块）。"
    )

    raw = client.retry_call(user_prompt, system_prompt, temperature=temperature)
    # 健壮解析：先直接 JSON 解析，失败再用正则提取第一个 JSON 数组
    decisions = []
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, list):
                decisions = parsed
        except Exception:
            try:
                m = _re.search(r"\[.*\]", raw, flags=_re.DOTALL)
                if m:
                    parsed = _json.loads(m.group(0))
                    if isinstance(parsed, list):
                        decisions = parsed
            except Exception:
                decisions = []

    cleaned = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        idx = d.get("index", None)
        score = d.get("score", 0)
        decision = d.get("decision", "drop")
        reason = d.get("reason", "")
        if isinstance(idx, int) and 0 <= idx < len(papers):
            try:
                score_val = float(score)
            except Exception:
                score_val = 0.0
            cleaned.append({"index": idx, "score": score_val, "decision": decision, "reason": reason})

    kept = [x for x in cleaned if x.get("decision") == "keep"]
    kept.sort(key=lambda x: x["score"], reverse=True)
    # 分数阈值过滤（若提供）
    if min_score is None:
        min_score = MIN_SCORE
    if min_score is not None:
        kept = [x for x in kept if x["score"] >= float(min_score)]
    # 不强制数量：默认保留全部 keep；若指定 top_k 则截断
    top = kept if top_k is None else kept[:max(0, min(top_k, len(kept)))]

    if not top:
        # 回退：若模型未选出任何 keep，则保留最多 3 篇作为兜底
        fallback_n = min(2, len(papers))
        top = [{"index": i, "score": 6.0, "decision": "keep", "reason": "回退策略"} for i in range(fallback_n)]

    selected = []
    for x in top:
        p = dict(papers[x["index"]])
        p["score"] = x["score"]
        p["decision"] = x["decision"]
        p["reason"] = x["reason"]
        selected.append(p)

    return selected

        
def cronjob():

    if SERVERCHAN_API_KEY is None:
        raise Exception("未设置SERVERCHAN_API_KEY环境变量")

    print('[+] 开始执行每日推送任务....')

    yesterday = get_yesterday()
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    print('[+] 开始检索每日最新论文....')
    papers = search_arxiv_papers(QUERY, LIMITS)
    # 使用大模型打分筛选（不强制数量，支持 MIN_SCORE 环境变量阈值）
    papers = rank_papers_with_deepseek(papers, top_k=None, min_score=MIN_SCORE)

    if papers == []:
        
        push_title = f'Arxiv:{QUERY}[X]@{today}'
        send_wechat_message('', '[WARN] NO UPDATE TODAY!', SERVERCHAN_API_KEY)

        print('[+] 每日推送任务执行结束')

        return True
        

    print('[+] 开始推送每日最新论文....')

    for ii, paper in enumerate(tqdm(papers, total=len(papers), desc=f"论文推送进度")):

        title = paper['title']
        url = paper['url']
        pub_date = paper['pub_date']
        # summary = paper['summary']
        translated = paper['translated']
        score = paper.get('score', 0)
        decision = paper.get('decision', 'unknown')
        reason = paper.get('reason', '')

        yesterday = get_yesterday()

        if pub_date == yesterday:
            msg_title = f'[Newest]{title}' 
        else:
            msg_title = f'{title}'

        msg_url = f'URL: {url}'
        msg_pub_date = f'Pub Date：{pub_date}'
        msg_rating = f'AI评分：{score:.1f}/10 | 决策：{decision}'
        if reason:
            msg_rating += f'\n理由：{reason}'
        # msg_summary = f'Summary：\n{summary}'
        msg_translated = f'Translated (Powered by {MODEL_TYPE}):\n{translated}'

        push_title = f'Arxiv:{QUERY}[{ii}]@{today}'
        msg_content = f"[{msg_title}]({url})\n{msg_pub_date}\n{msg_url}\n{msg_rating}\n{msg_translated}\n"

        # send_wechat_message(push_title, msg_content, SERVERCHAN_API_KEY)
        send_feishu_message(push_title, msg_content, FEISHU_URL)

        time.sleep(15)

    print('[+] 每日推送任务执行结束')

    return True


if __name__ == '__main__':
    cronjob()



