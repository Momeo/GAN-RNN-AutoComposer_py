import midi
import numpy as np
import os


class MidiConnector:

    lower_bound = 24
    upper_bound = 102
    span = upper_bound - lower_bound

    # midi文件转Note(音符)
    @staticmethod
    def midiToNoteStateMatrix(midi_file_path, span=span):
        pattern = midi.read_midifile(midi_file_path)

        time_left = []
        for track in pattern:
            time_left.append(track[0].tick)

        posns = [0 for track in pattern]

        statematrix = []
        time = 0

        state = [[0, 0] for x in range(span)]
        statematrix.append(state)
        condition = True
        while condition:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                oldstate = state
                state = [[oldstate[x][0], 0] for x in range(span)]
                statematrix.append(state)
            for i in range(len(time_left)):
                if not condition:
                    break
                while time_left[i] == 0:
                    track = pattern[i]
                    pos = posns[i]

                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < MidiConnector.lower_bound) or (evt.pitch >= MidiConnector.upper_bound):
                            pass
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch - MidiConnector.lower_bound] = [0, 0]
                            else:
                                state[evt.pitch - MidiConnector.lower_bound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            out = statematrix
                            condition = False
                            break
                    try:
                        time_left[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        time_left[i] = None

                if time_left[i] is not None:
                    time_left[i] -= 1

            if all(t is None for t in time_left):
                break

            time += 1

        S = np.array(statematrix)
        statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
        statematrix = np.asarray(statematrix).tolist()
        return statematrix

    # Note转midi文件
    @staticmethod
    def noteStateMatrixToMidi(statematrix, filename="output_file", span=span):
        statematrix = np.array(statematrix)
        if not len(statematrix.shape) == 3:
            statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
        statematrix = np.asarray(statematrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)

        span = MidiConnector.upper_bound - MidiConnector.lower_bound
        tickscale = 55

        lastcmdtime = 0
        prevstate = [[0, 0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + MidiConnector.lower_bound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=40, pitch=note + MidiConnector.lower_bound))
                lastcmdtime = time

            prevstate = state

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        midi.write_midifile("{}.mid".format(filename), pattern)

    # 读取midi数据
    @staticmethod
    def get_songs(midi_path):
        files = os.listdir(midi_path)
        songs = []
        for f in files:
            f = midi_path + '/' + f
            print('加载:', f)
            try:
                song = np.array(MidiConnector.midiToNoteStateMatrix(f))
                if np.array(song).shape[0] > 64:
                    songs.append(song)
            except Exception as e:
                print('数据无效: ', e)
        print("读取的有效midi文件个数: ", len(songs))
        return songs

