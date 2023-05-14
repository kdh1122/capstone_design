import flask
from flask import Flask, render_template, request
import sys
sys.path.append('/path/to/dash')

from dash import Dash
import numpy as np
import tensorflow as tf
#import dash_html_components as html
from dash import html
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import pandas as pd
from pathlib import Path
from dash import dcc
#import dash_core_components as dcc
import dash_bootstrap_components as dbc
import copy
from dash.dependencies import ClientsideFunction, Input, Output
import plotly.graph_objs as go
import datetime as dt


server = Flask(__name__)
dash_app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], server=server,
                url_base_pathname='/analysis/')
dash_app2 = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], server=server,
                 url_base_pathname='/information/')

PATH = Path(__file__).parent
DATA_PATH = PATH.joinpath("static").resolve()

df = pd.read_csv(DATA_PATH.joinpath("csv/data.csv"), low_memory=False, encoding="utf8")
df['report_time'] = pd.to_datetime(df['report_time'])
df_pie = pd.read_csv(DATA_PATH.joinpath("csv/pieplot.csv"), low_memory=False, encoding="utf8")
df_rank = pd.read_csv(DATA_PATH.joinpath("csv/score_data.csv"), low_memory=False, encoding="utf8")
df_rank2 = pd.read_csv(DATA_PATH.joinpath("csv/score_data2.csv"), low_memory=False, encoding="utf8")
df_mountains = pd.read_csv(DATA_PATH.joinpath("csv/mountain_weight.csv"), low_memory=False, index_col=0)

loc_statuses_dict = dict(서울특별시="서울특별시", 강원도="강원도", 경상남도="경상남도",
                         경상북도="경상북도", 광주광역시="광주광역시", 대구광역시="대구광역시",
                         대전광역시="대전광역시", 부산광역시="부산광역시", 세종특별자치시="세종특별자치시",
                         울산광역시="울산광역시", 인천광역시="인천광역시", 전라남도="전라남도",
                         전라북도="전라북도", 제주특별자치도="제주특별자치도", 충청남도="충청남도", 충청북도="충청북도",
                         )

loc_statuses_options = [    
    {'label': str(loc_statuses_dict[loc_status]), "value": str(loc_status), 'style': {'color': '#FFFFFF'}} for loc_status in loc_statuses_dict
]



mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"

data1 = [go.Pie(
    labels=['일반산악사고', '일반조난', '실족추락', '개인질환', '탈진.탈수', '자살기도', '암벽등반', '저체온증', '낙석.낙빙',
            '고온환경질환', '야생식물섭취중독'],
    values=[38996, 20760, 16270, 7846, 1896, 487, 311, 153, 146, 37, 26],
    name='사고원인'
)]

data2 = [go.Bar(
    x=list(df_pie['시간']),
    y=list(df_pie['사고건수']),
    name='시간별 사고 추이',
)]

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#59617e", #연보라색
    paper_bgcolor="#59617e", #남색
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    xaxis=dict(
        tickfont=dict(color='white'),
    ),
    yaxis=dict(
        tickfont=dict(color='white'),
    ),
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
    titlefont=dict(
        color='white'
    ),
)

layout1 = go.Layout(
    plot_bgcolor="#59617e",
    paper_bgcolor="#59617e",
    xaxis=dict(
        tickfont=dict(color='white'),
    ),
    yaxis=dict(
        tickfont=dict(color='white'),
    ),
    legend=dict(
        font=dict(color='white'),
    ),
)


def filter_dataframe(df, loc_statuses, year_slider):
    dff = df[
        df["산지역"].isin(loc_statuses)
        & (df["report_time"] > dt.datetime(year_slider[0], 1, 1))
        & (df["report_time"] < dt.datetime(year_slider[1], 12, 31))
        ]
    return dff


dash_app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "Prediction of Risk for Mountain Accidents",
                                    style={"margin-bottom": "0px",
                                           "color":"#FFFFFF"},
                                    
                                ),
                                html.H5(
                                    "Data Analysis Overview", style={"margin-top": "0px", "color":"#FFFFFF"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.Img(
                            src="/static/logo1-1.png",
                            id="plotly-image",
                            style={
                                "height": "130px",
                                "width": "auto",
                                "margin-bottom": "10px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                
                html.Div(
                    [
                        html.A(
                            html.Button("Home", id="learn-more-button1"),
                            href="/",
                        ),
                        html.A(
                            html.Button("Information", id="learn-more-button2"),
                            href="/information",
                        ),
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "보고싶은 년도 범위를 선택하세요(히스토그램에 표시할 범위를 선택하세요):",
                            className="control_label",
                            style={"color":"#FFFFFF"}
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=2010,
                            max=2154,
                            step=1,
                            marks={2010: 2010, 2022: 2011, 2034: 2012, 2046: 2013, 2058: 2014, 2070: 2015, 2082: 2016,
                                   2094: 2017, 2106: 2018, 2118: 2019, 2130: 2020, 2142: 2021},
                            value=[2010, 2154],
                            className="dcc_control",
                            
                        ),
                        html.P("전국 17개 시도:", className="control_label", style={'padding-top': "60px", "color":"#FFFFFF"}),

                        dcc.Dropdown(
                            id="loc_statuses",
                            options=loc_statuses_options,
                            multi=True,
                            value=list(loc_statuses_dict.keys()),
                            className="dcc_control",
                            style={'backgroundColor': '#59617e', 'color': 'black', 'border': '2px solid #FFFFFF'},
                        ),

                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(id="count_graph")],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
            style={"height": "auto"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("사고종별 산악사고 발생 비율", style={"color":"#FFFFFF"}),
                        dcc.Graph(
                            id="graph",
                            figure=go.Figure(data=data1, layout=layout1)
                        )
                    ],
                    className="pretty_container five columns",
                ),
                html.Div(
                    [
                        html.Div(
                            [html.H6("Mountain Accident", style={"color":"#FFFFFF"}),
                             html.H6("Mostly Occurred At", style={"color":"#FFFFFF"}),
                             html.H3("02:00 PM ", style={"color":"#FFFFFF"}),
                             html.H6("Top 3 Accident Frequent :", style={"color":"#FFFFFF"}),
                             html.H6("Mountains Are :", style={"color":"#FFFFFF"}),
                             html.H3("도봉산, 오봉산, 설악산", style={"color":"#FFFFFF"})
                             ],

                            id="wells",
                            className="mini_container",
                        ),
                    ],
                    className="pretty_container four columns"
                ),
                html.Div(
                    [
                        html.H4("시간대별 산악 사고 발생 건수", style={"color":"#FFFFFF"}),
                        dcc.Graph(
                            id="main_graph",
                            figure=go.Figure(data=data2, layout=layout1)
                        )],
                    className="pretty_container six columns",
                ),

            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dbc.ListGroup([
                            html.H4("최근 산악 사고 관련 기사", style={"color":"#FFFFFF"}),
                            dbc.ListGroupItem(
                                "강릉서 폭포에 빠진 60대 산악회원 숨져",
                                href="https://www.yna.co.kr/view/AKR20190820126100062?input=1195m",
                                className="row pretty_container columns",
                            ),
                            dbc.ListGroupItem(
                                "“은누리야”…폭염속 애타는 외침",
                                href="http://www.ccdn.co.kr/news/articleView.html?idxno=587862#09SX",
                                className="row pretty_container columns"
                            ),
                            dbc.ListGroupItem(
                                "[기고]여름철 산행 안전사고 주의해야",
                                href="http://www.asiatoday.co.kr/view.php?key=20190820010010413",
                                className="row pretty_container columns"
                            ),
                            dbc.ListGroupItem(
                                "60대 산악회 회원, 구룡폭포 계곡서 추락사",
                                href="http://cnews.getnews.co.kr/view.php?ud=20190818212014101988c3409001_16",
                                className="row pretty_container columns"
                            ),
                            dbc.ListGroupItem(
                                "여름철 지리산 산악사고 이렇게 예방하자",
                                href="http://www.gnmaeil.com/news/articleView.html?idxno=423562",
                                className="row pretty_container columns"
                            ),
                        ])
                    ], className="pretty_container five columns",
                ),
                html.Div(
                    [
                        html.Div(
                            html.H4('산악 사고 발생 빈도 TOP 300', style={"color":"#FFFFFF"})
                        ),
                        html.Div(
                            html.Iframe(
                                src=dash_app.get_asset_url("acct.html"),
                                style={
                                    "width": "100%",
                                    "height": "400px",
                                },
                            ),
                        ),
                    ],

                    className="pretty_container seven columns"
                ),

            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)
dash_app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)

dash_app2.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "Prediction of Risk for Mountain Accidents",
                                    style={"margin-bottom": "0px", "color":"#FFFFFF"},
                                ),
                                html.H5(
                                    "Data Analysis Overview", style={"margin-top": "0px", "color":"#FFFFFF"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.Img(
                            src="/static/logo1-1.png",
                            id="plotly-image",
                            style={
                                "height": "130px",
                                "width": "auto",
                                "margin-bottom": "10px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                
                html.Div(
                    [
                        html.A(
                            html.Button("Home", id="learn-more-button3"),
                            href="/",
                        ),
                        html.A(
                            html.Button("Analysis", id="learn-more-button4"),
                            href="/analysis",
                        ),
                    ],
                    className="one-third column",
                    id="button",
                ),

            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            html.H3('산악 사고 발생 장소', style={"color":"#FFFFFF"})
                        ),
                        html.Div(
                            html.Iframe(
                                src=dash_app2.get_asset_url("cluster.html"),
                                style={
                                    "width": "100%",
                                    "height": "400px",
                                },
                            ),
                        ),
                    ],
                    className="pretty_container seven columns"
                ),

                html.Div(
                    [
                        html.Div(
                            html.H3('사고 다발 지역 국가지점번호', style={"color":"#FFFFFF"}),
                        ),
                        html.Div(
                            [
                                html.Div([
                                    html.Img(
                                        src="/static/dobong.png",
                                        height="100px",
                                        width="100px",
                                        style={
                                            "float": "left",
                                        }
                                    ),
                                    html.Div([
                                        html.H6("도봉산 사고 다발 지역"),
                                        html.P("지점번호 : 다사 57XX 65YY"),
                                        html.P("상세지역 : 서울특별시 도봉구 도봉동 산29"),
                                    ], style={
                                        "padding-left": "150px",
                                        "color":"#FFFFFF",
                                    }
                                    ),
                                ], className="mini_container",
                                ),

                                html.Div([
                                    html.Img(
                                        src="/static/bukhan.png",
                                        height="100px",
                                        width="100px",
                                        style={
                                            "float": "left"
                                        }
                                    ),
                                    html.Div([
                                        html.H6("북한산 사고 다발 지역"),
                                        html.P("지점번호 : 다사 51XX 58YY"),
                                        html.P("상세지역 : 서울특별시 종로구 구기동 산1"),
                                    ], style={
                                        "padding-left": "150px",
                                        "color":"#FFFFFF",
                                    }
                                    ),
                                ], className="mini_container",
                                ),
                                html.Div([
                                    html.Img(
                                        src="/static/sulhak.png",
                                        height="100px",
                                        width="100px",
                                        style={
                                            "float": "left"
                                        }
                                    ),
                                    html.Div([
                                        html.H6("설악산 사고 다발 지역"),
                                        html.P("지점번호 : 다사 57XX 65YY"),
                                        html.P("상세지역 : 강원도 속초시 설악동 산31"),
                                    ], style={
                                        "padding-left": "150px",
                                        "color":"#FFFFFF",
                                    }
                                    ),
                                ], className="mini_container",
                                ),
                            ],
                        )
                    ],
                    className="pretty_container five columns"
                ),
            ],
            className="row flex-display",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            html.H3('Word Cloud', style={"color":"#FFFFFF"})
                        ),
                        html.Img(
                            title="사고 종별",
                            src="/static/word_cloud1.jpg",
                            id="word_cloud-image1",
                            style={
                                "height": "350px",
                                "width": "300px",
                            },
                            className="mini_container",
                        ),

                        html.Img(
                            title="산 별",
                            src="/static/word_cloud2.jpg",
                            id="word_cloud-image2",
                            style={
                                "height": "350px",
                                "width": "300px",
                            },
                            className="mini_container",
                        ),

                    ],
                    className="pretty_container seven columns"
                ),

                html.Div(
                    [
                        html.Div(
                            html.H3('영상', style={"color":"#FFFFFF"})
                        ),
                        html.Div(
                            html.Iframe(
                                src="https://www.youtube.com/embed/xR6gqTCf78c",
                                style={
                                    "width": "100%",
                                    "height": "400px",
                                },
                            ),
                        ),
                    ],
                    className="pretty_container five columns"
                ),
            ],
            className="row flex-display",
        )
    ],
    style={"display": "flex", "flex-direction": "column"},
)

# Slider -> count graph
@dash_app.callback(Output("year_slider", "value"), [Input("count_graph", "selectedData")])
def update_year_slider(count_graph_selected):
    if count_graph_selected is None:
        return [2010, 2154]

    nums = [int(point["pointNumber"]) for point in count_graph_selected["points"]]
    return [min(nums) + 2010, max(nums) + 2011]


@dash_app.callback(
    Output("count_graph", "figure"),
    [
        Input("loc_statuses", "value"),
        Input("year_slider", "value"),
    ],
)
def make_count_figure(loc_statuses, year_slider):
    layout_count = copy.deepcopy(layout)

    dff = filter_dataframe(df, loc_statuses, [2010, 2021])
    g = dff[["산지역", 'report_time']]
    g = g.set_index('report_time')
    g = g.resample("M").count()

    colors = []
    for i in range(2010, 2154):
        if i >= int(year_slider[0]) and i < int(year_slider[1]):
            colors.append("rgb(123, 199, 255)")
        else:
            colors.append("rgba(123, 199, 255, 0.2)")

    data = [
        dict(
            type="scatter",
            mode="markers",
            x=g.index,
            y=g["산지역"],
            name="All Wells",
            opacity=0,
            hoverinfo="skip",
        ),
        dict(
            type="bar",
            x=g.index,
            y=g["산지역"],
            name="bullshit",
            marker=dict(color=colors),
        ),
    ]

    layout_count["title"] = "시기별 산악사고 발생"
    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = False
    layout_count["autosize"] = True

    figure = dict(data=data, layout=layout_count)
    return figure


graph = tf.compat.v1.get_default_graph()

########################################################################################################################

@server.route('/')
def index():
    return render_template('index.html')


@server.route('/model')
def model():
    return render_template('tensor.html')


@server.route('/dash1')
def render_dash():
    return flask.redirect('/dash1')


@server.route('/dash2')
def render_dash1():
    return flask.redirect('/dash2')


app = DispatcherMiddleware(server, {
    '/dash1': dash_app.server,
    '/dash2': dash_app2.server
})

########################################################################################################################

@server.route('/weather', methods=['GET', 'POST'])
def get_post_javascript_data():
    max_list = request.form.get('max_list', 0)
    
    #모델 수정 부분, 원래는 6으로 돼 있었는데 9로 변경했어요! 저희 모델이 차원을 9개로 받더라구요. 여기서도 오류가 있었어요
    x = np.zeros((int(max_list), 9))
    for i in range(int(max_list)):
        x[i][0] = request.form.get(f'names[{i}][temp]', 0)
        x[i][1] = request.form.get(f'names[{i}][rain]', 0)
        x[i][2] = request.form.get(f'names[{i}][speed]', 0)
        x[i][3] = request.form.get(f'names[{i}][humidity]', 0)
        x[i][4] = request.form.get(f'names[{i}][dew]', 0)
        x[i][5] = request.form.get(f'names[{i}][snow]', 0)
    tf.keras.backend.clear_session()
    global graph
    with graph.as_default():
        model = tf.keras.models.load_model('my_model.h5')
        preds = model.predict(x)
    mountain_list = [request.form.get(f'mountains[{i}]', 0) for i in range(int(max_list))]
    
    # 사고가 발생한 데이터 수와 발생하지 않은 데이터 수
    n_accident = len(preds[preds[:, 1] >= 0.5])
    n_no_accident = len(preds[preds[:, 0] >= 0.5])

    # 데이터의 전체 수
    n_total = n_accident + n_no_accident

    # P(A|B) = P(B|A)P(A) / P(B)
    prior_prob = n_accident / n_total
    p_b_given_a = preds[:, 1].mean() # 예측 확률 벡터에서 사고 발생 확률 평균
    p_b_given_not_a = preds[:, 0].mean() # 예측 확률 벡터에서 사고 미발생 확률 평균

    # P(B) : 데이터가 이와 같이 나올 확률
    p_b = p_b_given_a * prior_prob + p_b_given_not_a * (1 - prior_prob)

    # P(A|B) : 사고가 발생할 확률(posterior probability)
    posterior_prob = p_b_given_a * prior_prob / p_b
    
    values2 = posterior_prob
    
    list_value = values2.tolist()
    list_weight = [df_mountains.loc[i, 'weight'] if i in df_mountains.index else 0.018298748185446165 for i in
                   mountain_list]
    result_array = np.array(list_value) * np.array(list_weight)
    minval = 0.00035494672381888403
    maxval = 41.753419663213286
    result_array = (result_array - minval) / (maxval - minval)
    temp_rk = df_rank["mns"].append(pd.Series(result_array), ignore_index=True).rank(pct=True, ascending=False)[-15:]
    return str(round(temp_rk * 100, 2).to_list()) #score_data에 mns로 저장

if __name__ == '__main__':
    app.run(debug=True,port=5000,host='localhost')