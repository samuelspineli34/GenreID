from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from midi2audio import FluidSynth
import librosa
import joblib
import mido
from collections import Counter
import music21 as m21
import time
import soundfile as sf
from werkzeug.utils import secure_filename
import shutil
import copy

app = Flask(__name__)
DIR_UPLOAD = "uploads"
EXT_PERMITIDAS = {'mid', 'mp3', 'wav', 'ogg', 'flac', 'm4a'}
path_base = os.path.dirname(os.path.realpath(__file__))
path_soundfont = os.path.join(path_base, "soundfont/GeneralUser-GS.sf2")

INSTRUMENTOS = {
    "classic": ["piano", "cordas", "flauta"],
    "country": ["violão", "percussão", "cordas"],
    "jazz": ["piano", "saxofone", "percussão"],
    "pop": ["piano", "violão", "sintetizador"],
    "rock": ["violão", "baixo", "percussão"]
}

MAPA_MIDI_INSTR = {
    'piano': 0,
    'violão': 24,
    'cordas': 48,
    'flauta': 73,
    'saxofone': 66,
    'percussão': 118,
    'baixo': 32,
    'sintetizador': 80
}

try:
    scaler = joblib.load(os.path.join(path_base, "modelos/scaler.pkl"))
    encoder = joblib.load(os.path.join(path_base, "modelos/encoder.pkl"))
    model = joblib.load(os.path.join(path_base, "modelos/random_forest_model.pkl"))
except Exception as e:
    print(f"ERROR: Falha ao carregar modelos: {str(e)}")

os.makedirs('static', exist_ok=True)
os.makedirs(DIR_UPLOAD, exist_ok=True)

def cria_padrao_instrumento(instrumento, acorde, tom, bpm):
    nota_quarto = 1.0
    nota_oitavo = 0.5
    nota_dezesseis = 0.25

    notas_acorde = list(acorde.pitches)
    if len(notas_acorde) < 3:
        raiz = acorde.root()
        if len(notas_acorde) == 1:
            notas_acorde.append(raiz.transpose(7))
            notas_acorde.append(raiz.transpose(12))
        elif len(notas_acorde) == 2:
            notas_acorde.append(raiz.transpose(12))

    notas_acorde_ajustadas = []
    for p in notas_acorde:
        if p.midi is not None and p.midi < 48:
            notas_acorde_ajustadas.append(p.transpose('octave'))
        elif p.midi is not None and p.midi > 84:
            notas_acorde_ajustadas.append(p.transpose('-octave'))
        else:
            notas_acorde_ajustadas.append(p)

    if instrumento == 'baixo':
        nota_raiz = acorde.root()
        if nota_raiz.midi is not None and nota_raiz.midi > 55:
            nota_raiz = nota_raiz.transpose('-octave')

        padrao = [
            (nota_raiz, nota_quarto, 90),
            (nota_raiz.transpose(7), nota_quarto, 90),
            (nota_raiz, nota_quarto, 90),
            (nota_raiz.transpose(7), nota_quarto, 90)
        ]
    elif instrumento == 'violão':
        padrao = [
            (acorde, nota_oitavo, 100),
            (None, nota_dezesseis, 0),
            (acorde, nota_dezesseis, 80),
            (None, nota_oitavo, 0)
        ]
    elif instrumento == 'piano':
        notas_acorde_baixo = sorted(acorde.pitches)[:3]
        acorde_baixo = m21.chord.Chord(notas_acorde_baixo)
        notas_melodia = sorted(acorde.pitches)[-3:]

        padrao = [
            (acorde_baixo, nota_oitavo, 100),
            (notas_melodia[0], nota_oitavo, 90),
            (notas_melodia[1], nota_dezesseis, 85),
            (notas_melodia[2], nota_dezesseis, 80)
        ]
    elif instrumento == 'percussão':
        padrao = [
            (m21.note.Note(midi=36), nota_dezesseis, 100),
            (m21.note.Note(midi=42), nota_dezesseis, 90),
            (m21.note.Note(midi=38), nota_dezesseis, 100),
            (m21.note.Note(midi=42), nota_dezesseis, 90),

            (m21.note.Note(midi=36), nota_dezesseis, 100),
            (m21.note.Note(midi=42), nota_dezesseis, 90),
            (m21.note.Note(midi=38), nota_dezesseis, 100),
            (m21.note.Note(midi=46), nota_dezesseis, 90)
        ]
    elif instrumento == 'sintetizador':
        padrao = [
            (acorde, nota_quarto * 2, 90),
            (None, nota_quarto, 0),
            (notas_acorde_ajustadas[0].transpose(12), nota_quarto, 80)
        ]
    elif instrumento == 'flauta':
        raiz_flauta = tom.pitchFromDegree(1).transpose(m21.interval.Interval('P12'))
        terca_flauta = tom.pitchFromDegree(3).transpose(m21.interval.Interval('P12'))
        quinta_flauta = tom.pitchFromDegree(5).transpose(m21.interval.Interval('P12'))

        if raiz_flauta.midi is not None and raiz_flauta.midi > 84:
            raiz_flauta = raiz_flauta.transpose('-octave')
            terca_flauta = terca_flauta.transpose('-octave')
            quinta_flauta = quinta_flauta.transpose('-octave')

        padrao = [
            (raiz_flauta, nota_oitavo, 80),
            (terca_flauta, nota_oitavo, 85),
            (quinta_flauta, nota_oitavo, 90),
            (None, nota_oitavo, 0)
        ]
    elif instrumento == 'cordas':
        padrao = [
            (acorde, nota_quarto * 4, 90)
        ]
    else:
        padrao = [
            (acorde, nota_quarto, 90),
            (acorde, nota_quarto, 90),
            (acorde, nota_quarto, 90),
            (acorde, nota_quarto, 90)
        ]
    return padrao


def obtem_obj_instrumento(nome_instrumento):
    nome_instrumento = nome_instrumento.lower()
    if nome_instrumento == 'piano':
        return m21.instrument.Piano()
    elif nome_instrumento == 'violão':
        return m21.instrument.AcousticGuitar()
    elif nome_instrumento == 'cordas':
        return m21.instrument.StringInstrument()
    elif nome_instrumento == 'flauta':
        return m21.instrument.Flute()
    elif nome_instrumento == 'saxofone':
        return m21.instrument.Saxophone()
    elif nome_instrumento == 'percussão':
        return m21.instrument.Percussion()
    elif nome_instrumento == 'baixo':
        return m21.instrument.AcousticBass()
    elif nome_instrumento == 'sintetizador':
        return m21.instrument.ElectricPiano()
    else:
        return m21.instrument.Piano()


@app.route('/generate_custom_accompaniment', methods=['POST'])
def gera_acompanhamento_personalizado():
    path_wav_acompanhamento = None
    path_midi = None
    path_mixado = None
    path_wav_orig_temp = None
    path_wav_orig_temp_dur = None
    dur_audio_orig_ms = 0

    try:
        dados = request.get_json()
        if not dados:
            return jsonify({'success': False, 'error': 'Dados não fornecidos'}), 400

        campos_obrigatorios = ['genre', 'scale', 'bpm', 'instrument', 'original_audio_path']
        if not all(campo in dados for campo in campos_obrigatorios):
            faltando = [f for f in campos_obrigatorios if f not in dados]
            return jsonify({'success': False, 'error': f'Campos obrigatórios faltando: {faltando}'}), 400

        genero = dados['genre'].lower()
        escala_str = dados['scale']
        bpm = int(dados['bpm'])
        instrumento = dados['instrument'].lower()
        nome_arq_audio_orig_static = dados['original_audio_path']

        if not escala_str or not isinstance(escala_str, str) or escala_str.strip() == '':
            return jsonify({'success': False, 'error': 'Erro: A string da escala está vazia ou inválida.'}), 400

        try:
            mapa_modo_escala_inv = {
                "maior": "major", "menor": "minor", "cromática": "chromatic",
                "dórico": "dorian", "frígio": "phrygian", "lídio": "lydian",
                "mixolídio": "mixolydian", "eólio": "aeolian", "lócrio": "locrian",
                "menor harmônica": "harmonic minor", "menor melódica": "melodic minor",
                "maior pentatônica": "pentatonic major", "menor pentatônica": "pentatonic minor",
            }
            mapa_tonica_inv = {
                'dó': 'C', 'dó sustenido': 'C#', 'ré bemol': 'D-',
                'ré': 'D', 'ré sustenido': 'D#', 'mi bemol': 'E-',
                'mi': 'E', 'fá': 'F', 'fá sustenido': 'F#',
                'sol bemol': 'G-', 'sol': 'G', 'sol sustenido': 'G#',
                'lá bemol': 'A-', 'lá': 'A', 'lá sustenido': 'A#',
                'si bemol': 'B-', 'si': 'B'
            }

            tonica_proc = 'C'
            modo_proc = 'major'

            escala_str_limpa = escala_str.lower().strip()

            if not escala_str_limpa:
                raise ValueError("String da escala vazia após limpeza.")

            ultima_palavra = escala_str_limpa.split(' ')[-1]
            if ultima_palavra in mapa_modo_escala_inv:
                modo_proc = mapa_modo_escala_inv[ultima_palavra]
                frase_tonica = ' '.join(escala_str_limpa.split(' ')[:-1]).strip()
            else:
                frase_tonica = escala_str_limpa
                if escala_str_limpa in mapa_modo_escala_inv:
                    modo_proc = mapa_modo_escala_inv[escala_str_limpa]
                    frase_tonica = ''

            if frase_tonica in mapa_tonica_inv:
                tonica_proc = mapa_tonica_inv[frase_tonica]
            elif frase_tonica == '':
                tonica_proc = 'C'
            else:
                achou_tonica = False
                for k, v in mapa_tonica_inv.items():
                    if k in frase_tonica:
                        tonica_proc = v
                        achou_tonica = True
                        break
                if not achou_tonica:
                    tonica_proc = 'C'

            if not tonica_proc:
                tonica_proc = 'C'

            tom = m21.key.Key(tonic=tonica_proc, mode=modo_proc)
        except Exception as e:
            print(f"ERROR: Falha no processamento da escala: {type(e).__name__}: {str(e)}")
            return jsonify({'success': False, 'error': f'Erro ao processar escala: {str(e)}'}), 400

        if instrumento not in MAPA_MIDI_INSTR:
            return jsonify({'success': False, 'error': f'Instrumento não suportado: {instrumento}'}), 400

        timestamp = str(int(time.time()))
        nome_arq_wav_acompanhamento = f"accompaniment_{timestamp}.wav"
        path_wav_acompanhamento = os.path.join('static', nome_arq_wav_acompanhamento)
        path_midi = os.path.join('static', f"temp_{timestamp}.mid")
        nome_arq_mixado = f"mixed_{timestamp}.mp3"
        path_mixado = os.path.join('static', nome_arq_mixado)

        path_completo_orig_static = os.path.join('static', nome_arq_audio_orig_static)
        nome_arq_audio_orig_frontend = nome_arq_audio_orig_static

        if not os.path.exists(path_completo_orig_static):
            return jsonify({'success': False,
                            'error': f'Arquivo original não encontrado no servidor: {nome_arq_audio_orig_static}'}), 400

        try:
            ext_orig = os.path.splitext(path_completo_orig_static)[1].lower()
            if ext_orig == '.mid':
                path_wav_orig_temp_dur = os.path.join('static',
                                                                   f"original_temp_for_duration_{timestamp}.wav")
                fs_temp = FluidSynth(path_soundfont)
                fs_temp.midi_to_audio(path_completo_orig_static, path_wav_orig_temp_dur)
                seg_audio_orig_dur = AudioSegment.from_wav(path_wav_orig_temp_dur)
                dur_audio_orig_ms = len(seg_audio_orig_dur)
                os.remove(path_wav_orig_temp_dur)
            else:
                seg_audio_orig = AudioSegment.from_file(path_completo_orig_static)
                dur_audio_orig_ms = len(seg_audio_orig)
        except Exception as e:
            print(
                f"WARNING: Não foi possível obter a duração do áudio original para ajustar o acompanhamento no início: {e}. Usando 30 segundos como padrão.")
            dur_audio_orig_ms = 30000
            if path_wav_orig_temp_dur and os.path.exists(path_wav_orig_temp_dur):
                os.remove(path_wav_orig_temp_dur)

        progressao = get_chord_progression(genero, tom)

        partitura = m21.stream.Score()

        inst = obtem_obj_instrumento(instrumento)
        parte = m21.stream.Part()
        parte.insert(0, inst)
        parte.insert(0, m21.tempo.MetronomeMark(number=bpm))

        min_dur_acompanhamento_seg = 5
        alvo_dur_acompanhamento_seg = max(min_dur_acompanhamento_seg, dur_audio_orig_ms / 1000)

        alvo_dur_ql = alvo_dur_acompanhamento_seg * (bpm / 60.0)

        dur_ql_atual = 0
        max_repeticoes = 100
        cont_repeticoes = 0

        while dur_ql_atual < alvo_dur_ql and cont_repeticoes < max_repeticoes:
            for graus_acorde in progressao:
                acorde_base = create_chord(tom, graus_acorde)
                padrao = cria_padrao_instrumento(instrumento, acorde_base, tom, bpm)

                for i, info_nota in enumerate(padrao):
                    nota_ou_acorde_original, duracao, velocidade = info_nota

                    if nota_ou_acorde_original is not None:
                        if isinstance(nota_ou_acorde_original, m21.pitch.Pitch):
                            n = m21.note.Note(nota_ou_acorde_original)
                            n.quarterLength = duracao
                            n.volume.velocity = velocidade
                            parte.append(n)
                        elif isinstance(nota_ou_acorde_original, m21.chord.Chord):
                            c = copy.deepcopy(nota_ou_acorde_original)
                            c.quarterLength = duracao
                            c.volume.velocity = velocidade
                            parte.append(c)
                        elif isinstance(nota_ou_acorde_original, m21.note.Note):
                            n = copy.deepcopy(nota_ou_acorde_original)
                            n.quarterLength = duracao
                            n.volume.velocity = velocidade
                            parte.append(n)
                    else:
                        r = m21.note.Rest()
                        r.quarterLength = duracao
                        parte.append(r)
                    dur_ql_atual += duracao
            cont_repeticoes += 1

        partitura.append(parte)

        partitura.write('midi', fp=path_midi)

        try:
            fs = FluidSynth(path_soundfont)
            fs.midi_to_audio(path_midi, path_wav_acompanhamento)
            if not os.path.exists(path_wav_acompanhamento):
                raise Exception("FluidSynth falhou ao criar o arquivo de acompanhamento WAV.")
        except Exception as e:
            print(f"ERROR: Erro GRAVE ao converter MIDI de acompanhamento para WAV: {type(e).__name__}: {str(e)}")
            if os.path.exists(path_midi):
                os.remove(path_midi)
            return jsonify({'success': False, 'error': f'Erro ao converter MIDI para WAV: {str(e)}'}), 500

        os.remove(path_midi)

        try:
            ext_orig = os.path.splitext(path_completo_orig_static)[1].lower()
            if ext_orig == '.mid':
                path_wav_orig_temp = os.path.join('static', f"original_temp_for_mix_{timestamp}.wav")
                fs.midi_to_audio(path_completo_orig_static, path_wav_orig_temp)
                audio_original = AudioSegment.from_wav(path_wav_orig_temp)
                nome_arq_audio_orig_frontend = os.path.basename(path_wav_orig_temp)
            else:
                audio_original = AudioSegment.from_file(path_completo_orig_static)

            audio_acompanhamento = AudioSegment.from_file(path_wav_acompanhamento)

            if len(audio_acompanhamento) < len(audio_original):
                num_repeticoes = int(np.ceil(len(audio_original) / len(audio_acompanhamento)))
                audio_acompanhamento = audio_acompanhamento * num_repeticoes

            audio_acompanhamento = audio_acompanhamento[:len(audio_original)]

            audio_mixado = (audio_original - 6).overlay(audio_acompanhamento)
            audio_mixado.export(path_mixado, format="mp3", bitrate="192k")

        except Exception as e:
            print(f"ERROR: Falha na etapa de carregamento/mixagem de áudios: {type(e).__name__}: {str(e)}")
            if os.path.exists(path_wav_acompanhamento):
                os.remove(path_wav_acompanhamento)
            if 'path_wav_orig_temp' in locals() and path_wav_orig_temp and os.path.exists(
                    path_wav_orig_temp):
                os.remove(path_wav_orig_temp)
            return jsonify({'success': False, 'error': f'Falha ao mixar áudios: {str(e)}'}), 500

        return jsonify({
            'success': True,
            'mixed_path': nome_arq_mixado,
            'original_path_for_playback': nome_arq_audio_orig_frontend,
            'accompaniment_path': nome_arq_wav_acompanhamento,
            'instrument': instrumento
        })

    except Exception as e:
        print(f"ERROR: Erro interno geral no servidor: {type(e).__name__}: {str(e)}")
        if 'path_wav_acompanhamento' in locals() and path_wav_acompanhamento and os.path.exists(path_wav_acompanhamento):
            os.remove(path_wav_acompanhamento)
        if 'path_midi' in locals() and path_midi and os.path.exists(path_midi):
            os.remove(path_midi)
        if 'path_wav_orig_temp' in locals() and path_wav_orig_temp and os.path.exists(path_wav_orig_temp):
            os.remove(path_wav_orig_temp)
        if 'path_wav_orig_temp_dur' in locals() and path_wav_orig_temp_dur and os.path.exists(
                path_wav_orig_temp_dur):
            os.remove(path_wav_orig_temp_dur)
        if 'path_mixado' in locals() and path_mixado and os.path.exists(path_mixado):
            os.remove(path_mixado)

        return jsonify({
            'success': False,
            'error': f'Erro interno do servidor: {str(e)}'
        }), 500


def get_chord_progression(genero, tom):
    progressoes = {
        'pop': [[1, 3, 5], [4, 6, 1], [5, 7, 2], [1, 3, 5]],
        'rock': [[1, 5], [4, 1], [5, 4], [1, 5]],
        'jazz': [[2, 5, 1], [6, 2, 5], [3, 6, 2], [1, 3, 6]],
        'classic': [[1, 4, 5], [6, 2, 3], [4, 1, 6], [5, 1, 4]],
        'country': [[1, 4, 5], [1, 5, 4], [1, 4, 1], [5, 1, 5]]
    }
    return progressoes.get(genero.lower(), [[1, 3, 5], [4, 6, 1], [5, 7, 2], [1, 3, 5]])


def create_chord(tom, graus):
    tons = [tom.pitchFromDegree(d) for d in graus]
    return m21.chord.Chord(tons)

TRAD_ESCALAS = {
    "major": "Maior",
    "minor": "Menor",
    "chromatic": "Cromática",
    "dorian": "Dórico",
    "phrygian": "Frígio",
    "lydian": "Lídio",
    "mixolydian": "Mixolídio",
    "aeolian": "Eólio",
    "locrian": "Lócrio",
    "harmonic minor": "Menor Harmônica",
    "melodic minor": "Menor Melódica",
    "pentatonic major": "Maior Pentatônica",
    "pentatonic minor": "Menor Pentatônica",
}


def traduz_nome_escala(nome_escala):
    partes = nome_escala.split(' ')
    if len(partes) > 1 and partes[-1].lower() in TRAD_ESCALAS:
        tonica = partes[0]
        modo = ' '.join(partes[1:]).lower()
        mapa_tonica = {
            'C': 'Dó', 'C#': 'Dó sustenido', 'D-': 'Ré bemol',
            'D': 'Ré', 'D#': 'Ré sustenido', 'E-': 'Mi bemol',
            'E': 'Mi', 'F': 'Fá', 'F#': 'Fá sustenido',
            'G-': 'Sol bemol', 'G': 'Sol', 'G#': 'Sol sustenido',
            'A-': 'Lá bemol', 'A': 'Lá', 'A#': 'Lá sustenido',
            'B-': 'Si bemol', 'B': 'Si'
        }
        tonica_traduzida = mapa_tonica.get(tonica, tonica)
        return f"{tonica_traduzida} {TRAD_ESCALAS[modo]}"

    return TRAD_ESCALAS.get(nome_escala.lower(), nome_escala)


@app.route('/predict', methods=['POST'])
def prediz():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo foi enviado.'}), 400

    arquivo = request.files['file']
    if arquivo.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado.'}), 400

    try:
        timestamp = str(int(time.time()))
        nome_arq_original = secure_filename(arquivo.filename)
        ext_arq = os.path.splitext(nome_arq_original)[1].lower()
        nome_arq_upload_unico = f"upload_{timestamp}{ext_arq}"
        path_arq = os.path.join(DIR_UPLOAD, nome_arq_upload_unico)
        arquivo.save(path_arq)

        nome_arq_static_unico = f"static_original_{timestamp}{ext_arq}"
        path_completo_orig_static = os.path.join('static', nome_arq_static_unico)
        shutil.copy(path_arq, path_completo_orig_static)

        if ext_arq == '.mid':
            notas = extrai_notas_midi(path_arq)
        else:
            notas = extrai_notas_audio(path_arq)

        obj_escala = determina_escala(notas)
        nome_escala = obj_escala.name if hasattr(obj_escala, 'name') else str(obj_escala)
        nome_escala_traduzido = traduz_nome_escala(nome_escala)

        bpm = detecta_bpm(path_arq)
        if isinstance(bpm, np.ndarray):
            bpm = bpm.item()
        bpm_arredondado = int(round(bpm))

        features = extrai_features(path_arq, bpm)

        features_escaladas = scaler.transform([features])

        idx_predito = model.predict(features_escaladas)[0]

        genero_predito = encoder.inverse_transform([idx_predito])[0]

        os.remove(path_arq)

        return jsonify({
            'predicted_genre': genero_predito,
            'predicted_scale': nome_escala_traduzido,
            'bpm': bpm_arredondado,
            'original_audio_path': nome_arq_static_unico,
            'available_instruments': INSTRUMENTOS.get(genero_predito.lower(), [])
        })

    except Exception as e:
        print(f"ERROR_PREDICT: Erro no processamento de predict: {str(e)}")
        if 'path_arq' in locals() and os.path.exists(path_arq):
            os.remove(path_arq)
        if 'path_completo_orig_static' in locals() and os.path.exists(path_completo_orig_static):
            os.remove(path_completo_orig_static)
        return jsonify({'error': str(e)}), 500


def eh_tipo_arquivo(nome_arquivo):
    return '.' in nome_arquivo and nome_arquivo.rsplit('.', 1)[1].lower() in EXT_PERMITIDAS


def extrai_notas_midi(arq_midi):
    mid = mido.MidiFile(arq_midi)
    notas = []
    for msg in mid:
        if msg.type == 'note_on' and msg.velocity > 0:
            notas.append(msg.note)
    return notas


def extrai_notas_audio(arq_audio):
    try:
        y, sr = librosa.load(arq_audio, duration=30, mono=True, sr=22050)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        soma_chroma = chroma.sum(axis=1)
        indices_notas = np.argsort(soma_chroma)[::-1][:5]

        notas_midi = [60 + i for i in indices_notas]
        return notas_midi
    except Exception as e:
        print(f"Erro ao extrair notas: {str(e)}")
        return []


def determina_escala(notas):
    if not notas:
        return m21.key.Key('C', 'major')

    fluxo = m21.stream.Stream()
    for nota_midi in notas:
        try:
            if 0 <= nota_midi <= 127:
                fluxo.append(m21.note.Note(midi=nota_midi))
            else:
                print(f"Nota MIDI inválida ignorada: {nota_midi}")
        except Exception as e:
            print(f"Erro ao criar nota music21: {e}, nota MIDI: {nota_midi}")
            continue

    if not fluxo.notes:
        return m21.key.Key('C', 'major')

    try:
        tom = fluxo.analyze('Krumhansl')
        return tom
    except Exception as e:
        print(f"Erro ao analisar escala: {str(e)}")
        return m21.key.Key('C', 'major')


def detecta_bpm(arq_audio):
    try:
        y, sr = librosa.load(arq_audio, duration=30, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except Exception as e:
        print(f"Erro ao detectar BPM: {str(e)}")
        return 120


def extrai_features(arq_audio, bpm):
    try:
        y, sr = librosa.load(arq_audio, duration=60, mono=True, sr=22050)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1)[:3],
            np.mean(zcr),
            bpm
        ])
        return features
    except Exception as e:
        print(f"Erro ao extrair features: {str(e)}")
        return np.zeros(13 + 3 + 1 + 1)


@app.route('/')
def index():
    return render_template('index.html', instrumentos=INSTRUMENTOS)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    os.makedirs(DIR_UPLOAD, exist_ok=True)
    app.run(debug=True)