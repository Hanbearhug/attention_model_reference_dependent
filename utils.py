import datetime

def get_past_window_feature(r, raw_data, lag_days, genre_type):
    """
    获取过去一段时间内的评分列表
    :param r:
    :param raw_data:
    :param lag_days:
    :param genre_type: 两种，total: 特定用户过去一段时间所有评分， same: 特定用户过去一段时间相同类型电影评分
    :return:
    """
    begin_date = r['date'] - datetime.timedelta(lag_days)
    end_date = r['date']

    if genre_type=='total':
        df_fea = raw_data.loc[(raw_data['userId'] == r['userId']) & (raw_data['date'] < end_date) &
                              (raw_data['date'] >= begin_date)]
    else:
        df_fea = raw_data.loc[(raw_data['userId'] == r['userId']) & (raw_data['date'] < end_date) &
                              (raw_data['date'] >= begin_date) & (raw_data[r['genres_list']].sum(axis=1) > 0)]
    if len(df_fea) ==0:
        return []
    else:
        return df_fea['rating'].tolist()

def get_past_movie_rating(r, raw_data, lag_days):
    """
    获得特定电影过去一段时间的评分均值
    :param r:
    :param raw_data:
    :param lag_days:
    :return:
    """
    begin_date = r['date'] - datetime.timedelta(lag_days)
    end_date = r['date']

    df_fea = raw_data.loc[(raw_data['movieId'] == r['movieId']) & (raw_data['date'] < end_date) &
                          (raw_data['date'] >= begin_date)]
    if len(df_fea) == 0:
        return None
    else:
        return df_fea['rating'].mean()