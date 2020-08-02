import datetime
import numpy as np

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

def get_agg_feature(r, feature_type, k=None):
    if len(r) == 0:
        return None
    r_list = [float(num) for num in r.split(',')]
    if feature_type == 'peak':
        return np.max(r_list)
    elif feature_type == 'end':
        return r_list[0]
    elif feature_type == 'avg':
        return np.mean(r_list)
    elif feature_type == 'sum':
        return np.sum(r_list)
    elif feature_type == 'top_k_avg':
        if len(r_list) <= k:
            return np.mean(r_list)
        else:
            np.mean(np.sort(r_list)[-k:])
    elif feature_type == 'end_k_avg':
        if len(r_list) <= k:
            return np.mean(r_list)
        else:
            np.mean(r_list[:k])
    else:
        raise ValueError("不存在的特征类型")

def past_score_feature(r, raw_data, genre_type, feature_type, lag_day, total_genres = total_movie_genres):
    """
    分各个类型的电影评价统计peak, sum, avg, end特征，如果genre为total，则不区分类型, 否则统计与当前电影具有相同特征的电影评分,
    此函数可以弃用，效率太低
    genre: 电影类型是否为同类型
    feature_type: peak、sum、avg、end
    lag_day: 过去的时间窗口，时间窗口为[begin_date, current_date) 注: current_date --> r['date']
    """
    begin_date = r['date'] - datetime.timedelta(lag_day)
    current_date = r['date']
    genres_list = [genre for genre in r['genres_list'] if genre in total_genres]
    """
    结合时间窗口和特征类型得到用于计算特征的表格
    """
    if feature_type == 'end':
        df_fea = raw_data.loc[(raw_data['userId']==r['userId']) & (raw_data['date']<current_date)]
    elif genre_type == 'total':
        df_fea = raw_data.loc[(raw_data['userId']==r['userId']) & (raw_data['date']<current_date) & (raw_data['date']>=begin_date)]
    elif genre_type == 'same':
        df_fea = raw_data.loc[(raw_data['userId']==r['userId']) & (raw_data['date']<current_date) &
                              (raw_data['date']>=begin_date) & (raw_data[genres_list].sum(axis=1)>0)]
    else:
        raise ValueError("没有这种类型的genre选择方式")
    """
    分特征类型计算
    """
    if len(df_fea)==0:
            return None
    else:
        if feature_type == 'end':
            df_fea.sort_values(['date'], ascending=False, inplace=True)
            df_fea.reset_index(drop=True, inplace=True)
            return df_fea.loc[0, 'rating']
        elif feature_type == 'peak':
            return df_fea['rating'].max()
        elif feature_type == 'avg':
            return df_fea['rating'].mean()
        elif feature_type == 'sum':
            return df_fea['rating'].sum()
        else:
            raise ValueError("不存在的特征类型")