package reversi;

import java.io.Serializable;
import java.util.AbstractMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by alexanderfedchin on 12/13/18.
 * This class represents is a Four-Layered hashmap (a hashmap of hashmaps of hashmaps of hashmaps).
 * This is an unelegant attempt to store 128 bit objects. It does work faster than a single
 * hash map, perhaps because I used an unefficient hash function when I tried the one-hashmap
 * solution
 */
public class FourLayeredHashMap implements Serializable {
    private AbstractMap<Integer, AbstractMap<Integer, AbstractMap<Integer, AbstractMap<Integer, Byte>>>> map;
    private long size;

    FourLayeredHashMap() {
        //TODO: What should initial capacity be
        map = new ConcurrentHashMap<>();
        size = 0;
    }

    public void putIfAbsent(int[] keys, Byte value) {
        //TODO: What should initial cpacity be
        AbstractMap<Integer, AbstractMap<Integer, AbstractMap<Integer, Byte>>> map2 =
                map.computeIfAbsent(keys[3], x -> new ConcurrentHashMap<>());
        AbstractMap<Integer,AbstractMap<Integer, Byte>> map3 = map2.computeIfAbsent(keys[2], x -> new ConcurrentHashMap<>());
        AbstractMap<Integer, Byte> map4 = map3.computeIfAbsent(keys[1], x -> new ConcurrentHashMap<>());
        if (map4.get(keys[0]) == null) {
            map4.put(keys[0], value);
            size += 1;
        }
    }

    public Byte get(int[] keys) {
        AbstractMap<Integer, AbstractMap<Integer, AbstractMap<Integer, Byte>>> map2 = map.get(keys[3]);
        if (map2 == null)
            return null;
        AbstractMap<Integer,AbstractMap<Integer, Byte>> map3 = map2.get(keys[2]);
        if (map3 == null)
            return null;
        AbstractMap<Integer, Byte> map4 = map3.get(keys[1]);
        if (map4 == null)
            return null;
        return map4.get(keys[0]);
    }

    public long size() {
        return size;
    }
}
