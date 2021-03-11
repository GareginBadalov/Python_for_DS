import pandas as pd

authors = pd.DataFrame({'author_id': [ 1, 2, 3],
                        'author_name':['Тургенев', 'Чехов', 'Островский']})
book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3],
                     'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
                     'price': [450, 300, 350, 500, 450, 370, 290]})
print(authors, '\n', book)
authors_price =pd.merge(authors, book, on='author_id', how='inner')
print(authors_price)

top5 = authors_price.sort_values(by='price', ascending=False).head()
print(top5)

my_min = authors_price.groupby('author_name').agg({'price': 'min'}).rename(columns={'price':'min_price'})
my_max = authors_price.groupby('author_name').agg({'price': 'max'}).rename(columns={'price':'max_price'})
my_mean = authors_price.groupby('author_name').agg({'price': 'mean'}).rename(columns={'price':'mean_price'})
authors_stat = pd.concat([my_min, my_max, my_mean], axis=1)
print(authors_stat)