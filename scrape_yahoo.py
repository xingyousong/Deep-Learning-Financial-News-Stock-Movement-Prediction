from bs4 import BeautifulSoup as Soup
import requests
import time

s = requests.Session()
res = s.get('https://news.search.yahoo.com/search;_ylt=?p=apple&fr=uh3_news_web_gs&fr2=piv-web&b=131')
if res.status_code != 200:
    raise Exception('failed request')
xml = res.text
soup = Soup(xml, 'lxml')
entries = soup.findAll('div', ['dd algo NewsArticle', 'dd algo fst NewsArticle', 'dd algo lst NewsArticle'])

print len(entries)
# print entries[0].contents
# print len(entries[0].contents)
print entries[0].contents[1].contents[0].contents[2].contents[0].contents[0]
# print entries[0].contents[1].contents[1].contents[0].contents
print entries[0].contents[1].contents[0].contents[0].contents[0].get('href')
# print entries[0].contents[1].contents[0].contents[0].contents[0].contents
# print ''.join(entries[0].contents[1].contents[0].contents[0].contents[0].contents)
# print ''.join([str(tp) for tp in entries[0].contents[1].contents[0].contents[0].contents[0].contents])


def remove_tags(s):
	in_tag = False
	tag_end = -1
	for i in range(len(s) - 1, -1, -1):
		if not in_tag and s[i] == '>':
			in_tag = True
			tag_end = i + 1
		elif in_tag and s[i] == '<':
			in_tag = False
			s = s[:i] + s[tag_end:]
	return s

print remove_tags(''.join([str(tp.encode('utf-8').decode('ascii', 'ignore')) for tp in entries[0].contents[1].contents[0].contents[0].contents[0].contents]))
# print [str(cp.encode('utf-8').decode('ascii', 'ignore')) for cp in entries[0].contents[1].contents[1].contents[0].contents]
print remove_tags(''.join([cp.encode('utf-8').decode('ascii', 'ignore') for cp in entries[0].contents[1].contents[1].contents[0].contents]))[:-4]

title = remove_tags(''.join([str(tp.encode('utf-8').decode('ascii', 'ignore')) for tp in entries[0].contents[1].contents[0].contents[0].contents[0].contents]))
article_link = entries[0].contents[1].contents[0].contents[0].contents[0].get('href')
news_source = entries[0].contents[1].contents[0].contents[2].contents[0].contents[0]
contents = remove_tags(''.join([cp.encode('utf-8').decode('ascii', 'ignore') for cp in entries[0].contents[1].contents[1].contents[0].contents]))[:-4]

res = s.get(article_link)
if res.status_code != 200:
    raise Exception('failed request')
xml = res.text

text = remove_tags(xml).encode('utf-8').decode('ascii', 'ignore')
# print text
contents_start = text.find(contents)
print text[contents_start:]
