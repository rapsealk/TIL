import java.rmi.registry.Registry;
import java.rmi.registry.LocateRegistry;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class Server extends Implementation {

    public Server() {

    }

    public static void main(String args[]) {
        try {
            // Instantiating the implementation class.
            Implementation obj = new Implementation();

            // Exporting the object of implementation class.
            // (here we are exporting the remote object to the stub)
            RemoteMethodInvocation stub = (RemoteMethodInvocation) UnicastRemoteObject.exportObject(obj, 0);

            // Binding the remote object (stub) in the registry.
            Registry registry = LocateRegistry.getRegistry();
            registry.bind("RemoteMethodInvocation", stub);
            System.out.println("Server ready");
        } catch (Exception e) {
            System.err.println("Server Exception: " + e.toString());
            e.printStackTrace();
        }
    }
}