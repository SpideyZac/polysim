use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use polysim::{
    physics::{PolyTrackPhysics, create_engine, create_linker},
    simulation::{PlayerController, PreparedTrack, SimulationWorker},
};

// Summer 1
const TRACK: &str = "PolyTrack24pdrVdskciFE8XCb3wRMNeeewNMvHNe2G3XftlmdkmRaPuRSkZdjoSysIWv4pl4S1STwDRzKiHJBJ44eVK0lkDs4GjdNUOdkl0FmEorcHZ02TdYD1uJUV6qgwRXKwjucnL0y81hxFFHx9yNOANRKFKNfS627RCOiwqLSqIoFlBca72GYWC0GAph3jAGQtbwLcAkYThEeuF9olyBEAFh0XtTDTYWzuHpCktWNeOnLnZuqGFmva4dRuBGoHED6dZNjEIIRZnQi3pt03tqWqgE5dojDp2nTLy0KJWBW7PRI69Ol7TOmESCvFQeAOFbrxfGDd3AbV9Au1jAqVdYY3W7YRJGbqEAj05u8aNQRzmVnPkJj1iljU51pCgrLeqSY2pUtaXYcKJNeIKQYs9T2LfGeQrsUYVLT25i1IU2N3Bl8l1leqWa9Ahmk9D3UmwcQNnSBlynlISO23aJaNFjeS5gW8IMsZRsfpcE4qbRabrXIDJBNlQy7t1BCfEjwGyXJt5vZKi4LMYIM4xTlsZZHXuz8k9QEeJs9gJpIv9MVynPlZt002SGkTjkEk3U8NkhGeJuPFiv0jNS91k0LET77Mr8ImufhDgY6cfjVPkwhV7MSaMR7Z1IQcesNq4wmIYcObELODkoehQuEC9mcr6ovy3p6aXdqvDXi2dtnRcbPAfYsve3RkAYafM3mK1vkMh37PuK1XIBQezZPwtmoeUfkigd4vXUuVvhifBxn4el3ZxffRWOLmHX06B9wQBrhr8Mr3xadI2F1aPEDngAqfOgzbbHRTneH6mRhxgIldZEyi8Wc5jd2gygS0y0desgpuAyAeNMmDcLcLie5B2lzyQ1zIHQQzWF1vJjmtnfbC6UIXy6yI5zGyGyPxa9gw8E3BU0sVT9VCSCdJXhtVQcW0SRbtBeI8nKHHXfKgve2NTrheuJ4wTUmGci0Na49RGxeHZE3fMjY6k9EjbtFPLjZdKvZm6BhORyuwuWOEM1EQqSISU7ext7lwBY4JdtTVjUhtxALC37eRG3HgcqviUavOESJ2eV2ZvkpRw8y6wejYq0wHN1xCErYWzOucVFPqVFYj4ATHrW5aZPzdm7sSMHv8WyeEVdJ8fN1lsmEm19fWvLhLf9lpyecfsJBbIsISLTt5HEQYfe9keEbFMVCy1L7lLywRopov0q76br3oM0WaQqp2Yf380jZHzEeGnkMqBqBivxoHWXHSSm38jeRzvt53NWecbE51qtjqeC6m0bvmRjnO8rHDfzXtNmd38iRhPGSn5Ib19VPzBeF5SFqB5gb6T2v0KsxReMyCc7YWaDQgWfzZd1kuuE7OWSkjYGPp4bfJZnu9KpkteZjpuqweFhY2FBqHxSyYxe7ZEYAepeBIIkMetGNhgeOxaNLjve8nNlcfGHD7FsGXhIkGHBJ0HvfAUfojDoO4Hne5oQn51oeLkEz2Efdke7OyE3SoCbk50cfrzvYsY8mMmcmsVeScqj2rOfS9epl5fpQFGqjhGHVXY83C4WEB6WTHjwueP3tYLCGCZiHyc0WeEcupH1Cps0mmikCQnnfvfKG12peXnT1XNIfZEoaEy9aAeQ6ciSb4Gm0pSyR1UFqclSEBfKq5MeIySoPFFiLRcK8mztlSDeujKhYkksZ28j57oG8nmFx2siyrewpyIEvczxnW2Unyqnlt7VqvNY4zoVCfKtrrBCb7w8zjheyXMAeqo6HTRwTOhjSBSoyaf9fQCgsPOdIuMfPgaGstZ";
const WASM: &str = "physics.wasm";

fn make_worker() -> SimulationWorker {
    let engine = create_engine();
    let (phys, _) = PolyTrackPhysics::from_file(&engine, WASM).unwrap();
    let prepared = PreparedTrack::from_export_string(TRACK).unwrap();
    let mut worker = SimulationWorker::new(phys, prepared).unwrap();
    worker.init().unwrap();
    worker
}

// How fast is one car for N frames
fn bench_single_car(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_car");

    for frames in [100u32, 500, 1000, 5000] {
        group.throughput(Throughput::Elements(frames as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(frames),
            &frames,
            |b, &frames| {
                let mut worker = make_worker();
                worker.create_car(0).unwrap();
                b.iter(|| {
                    for frame in 0..frames {
                        worker
                            .set_car_controls(
                                0,
                                PlayerController {
                                    up: frame > 10,
                                    ..Default::default()
                                },
                            )
                            .unwrap();
                        worker.update_car(0).unwrap();
                    }
                    // Reset for next iteration - reuses the car's state
                    // buffer instead of freeing and reallocating it.
                    worker.reset_car(0).unwrap();
                });
            },
        );
    }
    group.finish();
}

// How does throughput scale with number of simultaneous cars
fn bench_multi_car(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_car");
    const FRAMES: u32 = 500;

    for n_cars in [1usize, 2, 4, 8, 16] {
        // Throughput in total frames across all cars
        group.throughput(Throughput::Elements(n_cars as u64 * FRAMES as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_cars),
            &n_cars,
            |b, &n_cars| {
                let mut worker = make_worker();
                for id in 0..n_cars as u32 {
                    worker.create_car(id).unwrap();
                }

                b.iter(|| {
                    for frame in 0..FRAMES {
                        for id in 0..n_cars as u32 {
                            worker
                                .set_car_controls(
                                    id,
                                    PlayerController {
                                        up: frame > 10,
                                        ..Default::default()
                                    },
                                )
                                .unwrap();
                            worker.update_car(id).unwrap();
                        }
                    }
                    // Reset all cars - reuses each car's state buffer.
                    for id in 0..n_cars as u32 {
                        worker.reset_car(id).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

// Cost of create_car + delete_car vs. reset_car - matters if you're
// resetting frequently (brute-force search, TAS tooling).
fn bench_car_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("car_lifecycle");

    group.bench_function("create_delete", |b| {
        let mut worker = make_worker();
        b.iter(|| {
            worker.create_car(0).unwrap();
            worker.delete_car(0).unwrap();
        });
    });

    group.bench_function("reset", |b| {
        let mut worker = make_worker();
        worker.create_car(0).unwrap();
        b.iter(|| {
            worker.reset_car(0).unwrap();
        });
    });

    group.finish();
}

// Cost of PreparedTrack::from_export_string - should be done once
fn bench_track_decode(c: &mut Criterion) {
    c.bench_function("track_decode", |b| {
        b.iter(|| PreparedTrack::from_export_string(TRACK).unwrap());
    });
}

// Cost of spawning a worker: a fresh Linker per instance (from_module) vs.
// one Linker shared across every instantiation (from_module_with_linker).
// Matters if you spawn many short-lived workers, e.g. one per branch of a
// parallel search.
fn bench_worker_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_spawn");
    let engine = create_engine();
    let (_phys, module) = PolyTrackPhysics::from_file(&engine, WASM).unwrap();

    group.bench_function("fresh_linker_per_instance", |b| {
        b.iter(|| {
            PolyTrackPhysics::from_module(&engine, &module).unwrap();
        });
    });

    group.bench_function("shared_linker", |b| {
        let linker = create_linker(&engine).unwrap();
        b.iter(|| {
            PolyTrackPhysics::from_module_with_linker(&engine, &module, &linker).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_car,
    bench_multi_car,
    bench_car_lifecycle,
    bench_track_decode,
    bench_worker_spawn,
);
criterion_main!(benches);
