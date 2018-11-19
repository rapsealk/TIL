import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class Client {

    private Client() {

    }

    public static void main(String[] args) {
        try {
            // Getting the registry.
            Registry registry = LocateRegistry.getRegistry(null);

            // Looking up the registry for the remote object.
            RemoteMethodInvocation stub = (RemoteMethodInvocation) registry.lookup("RemoteMethodInvocation");

            // Calling the remote method using the obtained object.
            stub.printMessage();
        } catch (Exception e) {
            System.err.println("Client exception: " + e.toString());
            e.printStackTrace();
        }
    }
}