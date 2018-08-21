import io.reactivex.Observable;
import io.reactivex.disposables.Disposable;

public class FirstExample {

    public void emit() {
        Observable.just("Hello", "RxJava 2!!")
                // .subscribe(data -> System.out.println(data));
                .subscribe(System.out::println);
    }

    public static void main(String[] args) {
        // FirstExample demo = new FirstExample();
        // demo.emit();

        Disposable d = Observable.just(1,2,3,4,5,6,7,8,9,10)
                .subscribe(value -> {
                    System.out.println(
                            String.format("val: %d at %d", value*value*value, System.currentTimeMillis())
                    );
                });

        System.out.println("disposed: " + d.isDisposed() + " at " + System.currentTimeMillis());
    }
}
