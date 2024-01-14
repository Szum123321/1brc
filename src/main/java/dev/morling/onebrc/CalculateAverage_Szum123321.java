/*
 *  Copyright 2023 The original authors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package dev.morling.onebrc;

import jdk.incubator.vector.*;
import sun.misc.Unsafe;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;

public class CalculateAverage_Szum123321 {
    private static final String FILE = "./measurements.txt";

    // Primarly needed for the atomic operations, might use for faster memory access instead of the
    private static final Unsafe unsafe = getUnsafe();
    private static final boolean[][] MASK_CACHE;
    private static final VectorMask[] HASH_MASK_CACHE;

    private final static int STRING_BLOCK_SIZE = 102;
    private final static int STRING_BLOCK_COUNT = 20000;

    private static byte[] STATIC_STRING_STORAGE_BACKING = new byte[STRING_BLOCK_COUNT * STRING_BLOCK_SIZE];
    private static MemorySegment STATIC_STRING_STORAGE = MemorySegment.ofArray(STATIC_STRING_STORAGE_BACKING);// Arena.global().allocate(STRING_BLOCK_COUNT * STRING_BLOCK_SIZE);
    private static final AtomicInteger STATIC_STRING_OFFSET = new AtomicInteger(1);

    private static final int[] HASH_IV = {
            53987, 653, 3598307, 704897569, 1, 2, 3
    };

    private final static int HASH_MULT_CONSTANT = 0xB00F;
    private final static int HASH_ROT_CONST = 13;

    private static Vector<Integer> HASH_MULTIPLY_VECT = IntVector.SPECIES_128.broadcast(HASH_MULT_CONSTANT);

    /**
     * Layout:
     * <table class="tg">
     * <thead>
     *   <tr>
     *     <th class="tg-0pky">Field name</th>
     *     <th class="tg-0pky">type</th>
     *     <th class="tg-0pky">size</th>
     *     <th class="tg-0pky">offset</th>
     *     <th class="tg-0pky">description</th>
     *   </tr>
     * </thead>
     * <tbody>
     *   <tr>
     *     <td class="tg-0pky">Key pointer</td>
     *     <td class="tg-0pky">Pointer</td>
     *     <td class="tg-0pky">8</td>
     *     <td class="tg-0pky">0</td>
     *     <td class="tg-0pky">pointer into STATIC_STRING_STORAGE</td>
     *   </tr>
     *   <tr>
     *     <td class="tg-0pky">min</td>
     *     <td class="tg-0pky">int</td>
     *     <td class="tg-0pky">4</td>
     *     <td class="tg-0pky">8</td>
     *     <td class="tg-0pky">minimum accumulator</td>
     *   </tr>
     *   <tr>
     *     <td class="tg-0pky">max</td>
     *     <td class="tg-0pky">int</td>
     *     <td class="tg-0pky">4</td>
     *     <td class="tg-0pky">12</td>
     *     <td class="tg-0pky">max accumulator</td>
     *   </tr>
     *   <tr>
     *     <td class="tg-0pky">sum</td>
     *     <td class="tg-0pky">int</td>
     *     <td class="tg-0pky">4</td>
     *     <td class="tg-0pky">16</td>
     *     <td class="tg-0pky">sum of the values</td>
     *   </tr>
     *   <tr>
     *     <td class="tg-0pky">count</td>
     *     <td class="tg-0pky">int</td>
     *     <td class="tg-0pky">4</td>
     *     <td class="tg-0pky">20</td>
     *     <td class="tg-0pky">number of updates</td>
     *   </tr>
     * </tbody>
     * </table>
     */
    private static final long HASH_TABLE_ENTRY_LENGTH = 24;
    private static final long HASH_TABLE_BIT_DEPTH = 20;
    private static final long HASH_TABLE_ENTRY_COUNT = 1 << HASH_TABLE_BIT_DEPTH;
    private static final long HASH_TABLE_SUB_ENTRY_COUNT = 4;
    private static final long HASH_TABLE_SIZE = HASH_TABLE_ENTRY_LENGTH * HASH_TABLE_ENTRY_COUNT * HASH_TABLE_SUB_ENTRY_COUNT;

    private static final long HASH_TABLE_KEY_POINTER_OFFSET = 0;
    private static final long HASH_TABLE_MIN_OFFSET = 8;
    private static final long HASH_TABLE_MAX_OFFSET = 12;
    private static final long HASH_TABLE_SUM_OFFSET = 16;
    private static final long HASH_TABLE_COUNT_OFFSET = 20;

    private static final AtomicIntegerArray HASH_LOCK;
    private static final MemorySegment HASH_TABLE = Arena.global().allocate(HASH_TABLE_SIZE);
    private static final long HASH_TABLE_ADDRESS = HASH_TABLE.address();

    private static final long[] SORTING_POINTER_TABLE = new long[STRING_BLOCK_COUNT];
    private static final long[] SORTING_SORT_ACC_TABLE = new long[STRING_BLOCK_COUNT * 256];
    private static final short[] STRING_SORT_COUNT_ARRAY = new short[STRING_BLOCK_COUNT];

    private static String segment_to_string(MemorySegment segment) {
        return new String(segment.toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8);
    }

    private static HashSet<String> DEBUG_SET = new HashSet<>(10000);

    public static void main(String[] args) throws IOException, RuntimeException {
        try (var channel = FileChannel.open(Path.of(FILE), StandardOpenOption.READ)) {
            long fileSize = channel.size();
            // Add 512 padding bytes to allow out of bounds reads with Vectors
            var mapping = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, Arena.global());

            System.err.println("File ready");

            // Prepare segments
            final int N = Runtime.getRuntime().availableProcessors();

            long[] ranges = new long[N + 1];
            ranges[0] = 0;
            ranges[N] = fileSize;

            for (int i = 1; i < N; i++) {
                ranges[i] = i * (fileSize / N);
                while (ranges[i] < fileSize && mapping.get(ValueLayout.JAVA_BYTE, ranges[i]++) != (byte) '\n') ;
            }

            long t1 = System.nanoTime();
            // Run the data interpretation
            IntStream.range(0, N).parallel().forEach(v -> parse(mapping, ranges[v], ranges[v + 1]));
            //parse(mapping, 0, fileSize);
            long t2 = System.nanoTime();

            // Copy the addresses
            int cnt = 0;
            for (long i = 0; i < HASH_TABLE_SIZE; i += HASH_TABLE_ENTRY_LENGTH) {
                // long key = HASH_TABLE.get(ValueLayout.JAVA_LONG, i);
                long key = unsafe.getLong(HASH_TABLE.address() + i);
                if (key != 0)
                    SORTING_POINTER_TABLE[cnt++] = i;
            }

            // Radix sort
            // Iterate over all string positions from the least significant
            for (int i = STRING_BLOCK_SIZE - 1; i >= 0; i--) {
                Arrays.fill(STRING_SORT_COUNT_ARRAY, (short) 0); // Reset the count

                for (int j = 0; j < cnt; j++) {
                    // Get the pointer to the string
                    // long string_ptr = HASH_TABLE.get(ValueLayout.JAVA_LONG, SORTING_POINTER_TABLE[j]);//unsafe.getLong(HASH_TABLE.address() + SORTING_POINTER_TABLE[j]);
                    long string_ptr = unsafe.getLong(HASH_TABLE.address() + SORTING_POINTER_TABLE[j]);
                    // Get the i-th character of the string

                    // int bt = Byte.toUnsignedInt(STATIC_STRING_STORAGE.get(ValueLayout.JAVA_BYTE, string_ptr + i));
                    int bt = Byte.toUnsignedInt(unsafe.getByte(STATIC_STRING_STORAGE_BACKING, Unsafe.ARRAY_BYTE_BASE_OFFSET + string_ptr + i));
                    // Push the hash table entry pointer to the accumulation array
                    SORTING_SORT_ACC_TABLE[STRING_BLOCK_COUNT * bt + STRING_SORT_COUNT_ARRAY[bt]++] = SORTING_POINTER_TABLE[j];
                }

                int k = 0;

                for (int ch = 0; ch < 256; ch++) {
                    // Copy elements back to the sorting array
                    // System.arraycopy(SORTING_SORT_ACC_TABLE, STRING_BLOCK_COUNT * ch, SORTING_POINTER_TABLE, k, STRING_SORT_COUNT_ARRAY[ch]);
                    unsafe.copyMemory(SORTING_SORT_ACC_TABLE,
                            Unsafe.ARRAY_LONG_BASE_OFFSET + STRING_BLOCK_COUNT * ch * 8L,
                            SORTING_POINTER_TABLE,
                            Unsafe.ARRAY_LONG_BASE_OFFSET + k * 8L,
                            STRING_SORT_COUNT_ARRAY[ch] * 8);
                    k += STRING_SORT_COUNT_ARRAY[ch];
                }
            }

            long t4 = System.nanoTime();

            System.out.write('{');

            for (int i = 0; i < cnt; i++) {
                long pos = HASH_TABLE.address() + SORTING_POINTER_TABLE[i];
                long key = unsafe.getLong(pos);

                int count = unsafe.getInt(pos + HASH_TABLE_COUNT_OFFSET);
                int sum = unsafe.getInt(pos + HASH_TABLE_SUM_OFFSET);
                int min = unsafe.getInt(pos + HASH_TABLE_MIN_OFFSET);
                int max = unsafe.getInt(pos + HASH_TABLE_MAX_OFFSET);
                /*
                 * int count = HASH_TABLE.get(ValueLayout.JAVA_INT, SORTING_POINTER_TABLE[i] + HASH_TABLE_COUNT_OFFSET);
                 * int sum = HASH_TABLE.get(ValueLayout.JAVA_INT, SORTING_POINTER_TABLE[i] + HASH_TABLE_SUM_OFFSET);
                 * int min = HASH_TABLE.get(ValueLayout.JAVA_INT, SORTING_POINTER_TABLE[i] + HASH_TABLE_MIN_OFFSET);
                 * int max = HASH_TABLE.get(ValueLayout.JAVA_INT, SORTING_POINTER_TABLE[i] + HASH_TABLE_MAX_OFFSET);
                 */

                System.out.write(STATIC_STRING_STORAGE_BACKING, (int) key, (int) get_c_str_length(STATIC_STRING_STORAGE.asSlice(key)));

                // System.out.printf("=%.1f/%.1f/%.1f/%d", min / 10.0, sum/(float)count/10, max/10.0, count);
                System.out.printf("=%.1f/%.1f/%.1f", min / 10.0, sum / (float) count / 10, max / 10.0);
                if (i < cnt - 1)
                    System.out.print(", ");
            }

            System.out.println('}');

            long t5 = System.nanoTime();

            System.err.printf("File processing: %f, Sorting: %f, Printing: %f\n", (t2 - t1) / 1000000.0f, (t4 - t2) / 1000000.0f, (t5 - t4) / 1000000.0f);
            System.err.printf("Entries: %d\n", cnt);
            int k = 0;
            for (int i = 0; i < HASH_TABLE_ENTRY_COUNT; i++)
                if (HASH_LOCK.get(i) != 1) {
                    k = Math.max(k, HASH_LOCK.get(i));
                }
            System.err.printf("Max hash hit ratio: %d\n", k);
            System.err.printf("Hash busy waiting counter: %d\n", COLLISION_COUNTER.get());
        }
    }

    private static void parse(MemorySegment segment, final long begin, final long end) {
        long i = begin;

        while (i < end) {
            long colon_offset = 0, hash;

            // VectorMask<Byte> colon_comp_res;
            int colon_comp_pos;
            var hash_state = IntVector.SPECIES_128.fromArray(HASH_IV, 0);

            do {
                var load_mask = ByteVector.SPECIES_128.indexInRange(i + colon_offset, end);
                var vec = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, segment, i + colon_offset, ByteOrder.LITTLE_ENDIAN, load_mask);

                colon_comp_pos = vec.compare(VectorOperators.EQ, (byte) ';').firstTrue();

                // var hash_mask = ByteVector.SPECIES_128.loadMask(MASK_CACHE[colon_comp_pos], 0);
                var hash_mask = (VectorMask<Byte>) HASH_MASK_CACHE[colon_comp_pos];

                hash_state = hash_state.lanewise(VectorOperators.XOR, vec.blend(0, hash_mask).reinterpretAsInts());
                hash_state = hash_round(hash_state);
                colon_offset += colon_comp_pos;
            } while (colon_comp_pos == 16);

            if (colon_offset >= end)
                return;

            // Unpacking the vector by hand is somehow much faster
            hash = hash_state.reduceLanesToLong(VectorOperators.XOR);

            hash ^= colon_offset;
            // hash *= HASH_MULT_CONSTANT; // finish round
            hash ^= (hash >> (31 - HASH_TABLE_BIT_DEPTH)) * HASH_MULT_CONSTANT;
            hash &= (1 << HASH_TABLE_BIT_DEPTH) - 1; // Truncate down to the desired bit length

            long j = 1;
            byte byte_val;
            int value = 0, sign = 1;

            if ((byte_val = segment.get(ValueLayout.JAVA_BYTE, i + colon_offset + j)) == '-')
                sign = -1;
            else
                value = byte_val - '0';

            j++;

            // while((byte_val = segment.get(ValueLayout.JAVA_BYTE, i + colon_offset + j)) != '\n' && byte_val != 0) {
            while ((byte_val = unsafe.getByte(segment.address() + i + colon_offset + j)) != '\n' && byte_val != 0) {
                if (byte_val != '.') {
                    value *= 10;
                    value += byte_val - '0';
                }
                j++;
            }

            value *= sign;

            // The memory address of the hash table subcell
            long table_entry = get_hash_table_entry((int) hash, segment.asSlice(i), (int) colon_offset);

            unsafe.getAndAddInt(null, table_entry + HASH_TABLE_SUM_OFFSET, value);
            unsafe.getAndAddInt(null, table_entry + HASH_TABLE_COUNT_OFFSET, 1);
            atomic_int_min(table_entry + HASH_TABLE_MIN_OFFSET, value);
            atomic_int_max(table_entry + HASH_TABLE_MAX_OFFSET, value);

            i += colon_offset + j + 1;
        }
    }

    private final static AtomicInteger COLLISION_COUNTER = new AtomicInteger(0);

    private static long get_hash_table_entry(final int hash, final MemorySegment raw_string, int key_length) {
        // Pointer to the first entry matching this hash
        // final long hash_table_base_offset = /*HASH_TABLE.address() +*/ hash * HASH_TABLE_ENTRY_LENGTH * HASH_TABLE_SUB_ENTRY_COUNT;
        final long hash_table_base_offset = HASH_TABLE.address() + hash * HASH_TABLE_ENTRY_LENGTH * HASH_TABLE_SUB_ENTRY_COUNT;

        while (true) {
            // Reset the counters
            long hash_entry_ptr = hash_table_base_offset;

            // Lock and available_access_counter store the number of valid entries + 1, or are negative if the table is being modified right now.
            int lock, available_access_counter;

            // Busy wait while the hash table is being updated
            //while ((available_access_counter = lock = HASH_LOCK.get(hash)) < 0) ;
            available_access_counter = Math.abs(lock = HASH_LOCK.get(hash));
            // Get the key_pointer at hash_entry_ptr
            // long name_pointer = HASH_TABLE.get(ValueLayout.JAVA_LONG, hash_entry_ptr);
            long name_pointer = unsafe.getLong(hash_entry_ptr);

            // Advance as long as there are new entries left and the string pointed to by name_pointer is different to raw_string
            while (--available_access_counter > 0 &&
            // simd_string_compare(raw_string, key_length, STATIC_STRING_STORAGE.asSlice(name_pointer)) != 0) {
             //simd_string_compare(raw_string.asSlice(0, key_length), STATIC_STRING_STORAGE.asSlice(name_pointer)) != 0) {
                    !compare_key_with_string_stack(raw_string, key_length, name_pointer)) {
                hash_entry_ptr += HASH_TABLE_ENTRY_LENGTH;
                // name_pointer = HASH_TABLE.get(ValueLayout.JAVA_LONG, hash_entry_ptr);
                name_pointer = unsafe.getLong(hash_entry_ptr);
            }

            if (available_access_counter != 0 /* && name_pointer != 0 */) {
                // The entry is valid and matches
                return hash_entry_ptr;
            }
            else if(lock > 0) {
                // Try to claim this table supercell by negating the lock. if the lock value has changed in the meantime, restart the procedure
                if (HASH_LOCK.compareAndSet(hash, lock, -lock)) {
                    // We've successfully locked the table
                    // Push the new string to the key name area
                    long new_string_ptr = push_new_string(raw_string.asSlice(0, key_length));
                    // Store the pointer in the array
                    // HASH_TABLE.set(ValueLayout.JAVA_LONG, hash_entry_ptr, new_string_ptr);
                    unsafe.putLong(hash_entry_ptr, new_string_ptr);
                    // Unlock the table and increment the lock counter
                    HASH_LOCK.set(hash, lock + 1);

                    return hash_entry_ptr;
                }
            }
        }
    }

    private static boolean compare_key_with_string_stack(final MemorySegment key, final long key_length, final long region_position) {
        final var SPECIES = ByteVector.SPECIES_PREFERRED;
        long i = 0;

        for (; i < SPECIES.loopBound(key_length); i += SPECIES.length()) {
            var x = ByteVector.fromMemorySegment(SPECIES, key, i, ByteOrder.LITTLE_ENDIAN);
            var y = ByteVector.fromMemorySegment(SPECIES, STATIC_STRING_STORAGE, region_position + i, ByteOrder.LITTLE_ENDIAN);
            if (!x.compare(VectorOperators.EQ, y).allTrue())
                return false;
        }

        for (; i < key_length; i++)
            if (unsafe.getByte(key.address() + i) != unsafe.getByte(STATIC_STRING_STORAGE_BACKING, Unsafe.ARRAY_BYTE_BASE_OFFSET + region_position + i))
                return false;

        return true;
    }

    /**
     * Checks if the bytes of key match region exactly
     * @param key first memory segment to compare
     * @param region region of the memory to test
     * @return the position of the mismatch + 1 or 0 on success
     */
    private static long simd_string_compare(MemorySegment key, MemorySegment region) {
        assert key.byteSize() <= region.byteSize();

        final var SPECIES = ByteVector.SPECIES_PREFERRED;
        long i = 0;

        for (; i < key.byteSize(); i += SPECIES.length()) {
            var mask = SPECIES.indexInRange(i, key.byteSize());
            var x = ByteVector.fromMemorySegment(SPECIES, key, i, ByteOrder.LITTLE_ENDIAN, mask);
            var y = ByteVector.fromMemorySegment(SPECIES, region, i, ByteOrder.LITTLE_ENDIAN, mask);
            var comp_result = x.compare(VectorOperators.EQ, y);

            if (!comp_result.allTrue())
                return i + comp_result.not().firstTrue() + 1;
        }

        return 0;
    }

    /*
     * private static long simd_string_compare(MemorySegment key, int key_length, MemorySegment region) {
     * assert key_length <= region.byteSize();
     * 
     * final var SPECIES = ByteVector.SPECIES_PREFERRED;
     * long i = 0;
     * 
     * for (; i < key_length; i += SPECIES.length()) {
     * var mask = SPECIES.indexInRange(i, key_length);
     * var x = ByteVector.fromMemorySegment(SPECIES, key, i, ByteOrder.LITTLE_ENDIAN, mask);
     * var y = ByteVector.fromMemorySegment(SPECIES, region, i, ByteOrder.LITTLE_ENDIAN, mask);
     * var comp_result = x.compare(VectorOperators.EQ, y);
     * 
     * if (!comp_result.allTrue())
     * return i + comp_result.not().firstTrue() + 1;
     * }
     * 
     * return 0;
     * }
     */

    private static long push_new_string(MemorySegment source_segment) {
        long dest_offset = STATIC_STRING_OFFSET.getAndAdd(STRING_BLOCK_SIZE);
        STATIC_STRING_STORAGE.asSlice(dest_offset).copyFrom(source_segment);
        return dest_offset;
    }

    private static long get_c_str_length(MemorySegment segment) {
        long i = 0;
        VectorMask<Byte> comp_res;
        do {
            var mask = ByteVector.SPECIES_PREFERRED.indexInRange(i, segment.byteSize());
            var vect = ByteVector.fromMemorySegment(ByteVector.SPECIES_PREFERRED, segment, i, ByteOrder.LITTLE_ENDIAN, mask);
            comp_res = vect.eq((byte) 0);
            i += comp_res.firstTrue();
        } while (!comp_res.anyTrue());
        return i;
    }

    private static void atomic_int_min(final long address, final int value) {
        int old, x;
        do {
            old = unsafe.getInt(null, address);
            x = Math.min(value, old);
        } while (!unsafe.compareAndSwapInt(null, address, old, x));
    }

    private static void atomic_int_max(final long address, final int value) {
        int old, x;
        do {
            old = unsafe.getInt(null, address);
            x = Math.max(value, old);
        } while (!unsafe.compareAndSwapInt(null, address, old, x));
    }

    private static Vector<Integer> hash_round(Vector<Integer> vec) {
        return vec.lanewise(VectorOperators.XOR, vec.lanewise(VectorOperators.ROL, HASH_ROT_CONST)).mul(HASH_MULTIPLY_VECT);
    }

    private static Unsafe getUnsafe() {
        try {
            Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
            theUnsafe.setAccessible(true);
            return (Unsafe) theUnsafe.get(Unsafe.class);
        }
        catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    static {
        HASH_MASK_CACHE = new VectorMask[17];

        for (int i = 0; i < 17; i++) {
            boolean[] a = new boolean[17];
            int j = 0;

            for (; j < i; j++)
                a[j] = false;
            for (; j < 17; j++)
                a[j] = true;
            HASH_MASK_CACHE[i] = ByteVector.SPECIES_128.loadMask(a, 0);
        }

        MASK_CACHE = new boolean[17][];
        for (int i = 0; i < 17; i++) {
            MASK_CACHE[i] = new boolean[17];
            int j = 0;

            for (; j < i; j++)
                MASK_CACHE[i][j] = false;
            for (; j < 17; j++)
                MASK_CACHE[i][j] = true;
        }

        int[] array = new int[(int) HASH_TABLE_ENTRY_COUNT];
        Arrays.fill(array, 1);
        HASH_LOCK = new AtomicIntegerArray(array);

        HASH_TABLE.fill((byte) 0);

        for (long i = 0; i < HASH_TABLE_ENTRY_COUNT * HASH_TABLE_SUB_ENTRY_COUNT * HASH_TABLE_ENTRY_LENGTH; i += HASH_TABLE_ENTRY_LENGTH) {
            HASH_TABLE.set(ValueLayout.JAVA_INT_UNALIGNED, i + HASH_TABLE_MIN_OFFSET, Integer.MAX_VALUE);
            HASH_TABLE.set(ValueLayout.JAVA_INT_UNALIGNED, i + HASH_TABLE_MAX_OFFSET, Integer.MIN_VALUE);
        }
    }
}
