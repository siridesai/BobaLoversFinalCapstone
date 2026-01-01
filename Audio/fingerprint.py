#make fingerprints for each peak in a list of peaks
def make_fingerprints(peaks, n_neighbors):
    #sort peaks by frequency and time
    peaks = sorted(peaks, key=lambda x: (x[0], x[1]))

    fingerprints = []

    for i, current_peak in enumerate(peaks): #for index i and current peak
        f1, t1 = current_peak
        count = 0 #0 neighbors found so far
        fanout = []

        #go through peaks after to find neighbors
        for peak in peaks[i + 1:]:
            f2, t2 = peak
            delta_t = t2 - t1

            if delta_t >= 0: #the neighbor happens after the peak
                fanout.append(((f1, f2, delta_t), t1)) #add the neighbor to fanout
                count += 1

            if count >= n_neighbors: #once all neighbors are found
                break
        
        fingerprints.append(fanout) #add fanout to fingerprints
    print("done")
    return fingerprints